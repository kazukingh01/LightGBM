/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_METRIC_CUSTOM_METRIC_HPP_
#define LIGHTGBM_SRC_METRIC_CUSTOM_METRIC_HPP_

#include "multiclass_metric.hpp"

namespace LightGBM {

/*! \brief Focal Loss metric for multiclass task */
class FocalLossMetric: public MulticlassMetric<FocalLossMetric> {
 public:
  explicit FocalLossMetric(const Config& config) : MulticlassMetric<FocalLossMetric>(config) {}

  inline static double LossOnPoint(label_t label, std::vector<double>* score, const Config& config) {
    double gamma = config.focal_gamma;
    size_t k = static_cast<size_t>(label);
    auto& ref_score = *score;
    double p_k = ref_score[k];
    if (p_k > kEpsilon) {
      // Focal loss: -((1 - p_k)^gamma) * log(p_k)
      return -std::pow(1.0 - p_k, gamma) * std::log(p_k);
    } else {
      return -std::pow(1.0 - kEpsilon, gamma) * std::log(kEpsilon);
    }
  }

  inline static const std::string Name(const Config&) {
    return "focalloss";
  }
};

/*! \brief XE-NDCG style metric for multiclass task with mixed-radix encoded labels */
class XendcgSoftmaxMetric: public Metric {
 public:
  explicit XendcgSoftmaxMetric(const Config& config) : config_(config) {
    num_class_ = config.num_class;
  }

  virtual ~XendcgSoftmaxMetric() {}

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return -1.0f;
  }

  /*!
   * \brief Decode a mixed-radix encoded value into a vector of digits
   * \param value The encoded integer value
   * \param num_class The radix for each position
   * \return Vector of decoded digits
   */
  static std::vector<int> MixedRadixDecode(int value, int num_class) {
    std::vector<int> digits;
    digits.reserve(num_class);
    for (int i = 0; i < num_class; ++i) {
      digits.push_back(value % num_class);
      value /= num_class;
    }
    return digits;
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    name_.emplace_back("multirank");
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();

    if (weights_ == nullptr) {
      sum_weights_ = static_cast<double>(num_data_);
    } else {
      sum_weights_ = 0.0;
      for (data_size_t i = 0; i < num_data_; ++i) {
        sum_weights_ += weights_[i];
      }
    }

    // Decode labels and compute phi (ideal distribution) for each data point
    label_int_2d_.resize(num_data_);
    phi_.resize(num_data_);
    for (data_size_t i = 0; i < num_data_; ++i) {
      int encoded_value = static_cast<int>(label_[i]);
      label_int_2d_[i] = MixedRadixDecode(encoded_value, num_class_);

      // Validate decoded labels
      for (int j = 0; j < num_class_; ++j) {
        if (label_int_2d_[i][j] < 0 || label_int_2d_[i][j] >= num_class_) {
          Log::Fatal("Decoded label must be in [0, %d), but found %d at position [%d][%d]",
                     num_class_, label_int_2d_[i][j], i, j);
        }
      }

      // Compute phi (ideal distribution based on relevance)
      phi_[i].resize(num_class_);
      for (int k = 0; k < num_class_; ++k) {
        phi_[i][k] = std::pow(2.0, num_class_ - label_int_2d_[i][k]);
      }
      // Softmax on phi
      double max_phi = *std::max_element(phi_[i].begin(), phi_[i].end());
      double sum_exp = 0.0;
      for (int k = 0; k < num_class_; ++k) {
        phi_[i][k] = std::exp(phi_[i][k] - max_phi);
        sum_exp += phi_[i][k];
      }
      for (int k = 0; k < num_class_; ++k) {
        phi_[i][k] /= sum_exp;
      }
    }
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override {
    double sum_loss = 0.0;
    int num_tree_per_iteration = num_class_;
    int num_pred_per_row = num_class_;
    if (objective != nullptr) {
      num_tree_per_iteration = objective->NumModelPerIteration();
      num_pred_per_row = objective->NumPredictOneRow();
    }

    if (objective != nullptr) {
      if (weights_ == nullptr) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          std::vector<double> raw_score(num_tree_per_iteration);
          for (int k = 0; k < num_tree_per_iteration; ++k) {
            size_t idx = static_cast<size_t>(num_data_) * k + i;
            raw_score[k] = static_cast<double>(score[idx]);
          }
          std::vector<double> rho(num_pred_per_row);
          objective->ConvertOutput(raw_score.data(), rho.data());
          // Compute cross-entropy loss: -sum_k(phi_k * log(rho_k))
          double loss = 0.0;
          for (int k = 0; k < num_class_; ++k) {
            double rho_k = std::max(rho[k], kEpsilon);
            loss -= phi_[i][k] * std::log(rho_k);
          }
          sum_loss += loss;
        }
      } else {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          std::vector<double> raw_score(num_tree_per_iteration);
          for (int k = 0; k < num_tree_per_iteration; ++k) {
            size_t idx = static_cast<size_t>(num_data_) * k + i;
            raw_score[k] = static_cast<double>(score[idx]);
          }
          std::vector<double> rho(num_pred_per_row);
          objective->ConvertOutput(raw_score.data(), rho.data());
          // Compute cross-entropy loss: -sum_k(phi_k * log(rho_k))
          double loss = 0.0;
          for (int k = 0; k < num_class_; ++k) {
            double rho_k = std::max(rho[k], kEpsilon);
            loss -= phi_[i][k] * std::log(rho_k);
          }
          sum_loss += loss * weights_[i];
        }
      }
    } else {
      if (weights_ == nullptr) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          std::vector<double> rho(num_tree_per_iteration);
          for (int k = 0; k < num_tree_per_iteration; ++k) {
            size_t idx = static_cast<size_t>(num_data_) * k + i;
            rho[k] = static_cast<double>(score[idx]);
          }
          // Compute cross-entropy loss: -sum_k(phi_k * log(rho_k))
          double loss = 0.0;
          for (int k = 0; k < num_class_; ++k) {
            double rho_k = std::max(rho[k], kEpsilon);
            loss -= phi_[i][k] * std::log(rho_k);
          }
          sum_loss += loss;
        }
      } else {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          std::vector<double> rho(num_tree_per_iteration);
          for (int k = 0; k < num_tree_per_iteration; ++k) {
            size_t idx = static_cast<size_t>(num_data_) * k + i;
            rho[k] = static_cast<double>(score[idx]);
          }
          // Compute cross-entropy loss: -sum_k(phi_k * log(rho_k))
          double loss = 0.0;
          for (int k = 0; k < num_class_; ++k) {
            double rho_k = std::max(rho[k], kEpsilon);
            loss -= phi_[i][k] * std::log(rho_k);
          }
          sum_loss += loss * weights_[i];
        }
      }
    }
    double loss = sum_loss / sum_weights_;
    return std::vector<double>(1, loss);
  }

 private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Pointer of weights */
  const label_t* weights_;
  /*! \brief Sum weights */
  double sum_weights_;
  /*! \brief Name of this metric */
  std::vector<std::string> name_;
  /*! \brief Number of classes */
  int num_class_;
  /*! \brief Config parameters */
  Config config_;
  /*! \brief Decoded labels: 2D array of shape (num_data_, num_class_) */
  std::vector<std::vector<int>> label_int_2d_;
  /*! \brief Pre-computed phi (ideal distribution) for each data point */
  std::vector<std::vector<double>> phi_;
};

}  // namespace LightGBM
#endif  // LIGHTGBM_SRC_METRIC_CUSTOM_METRIC_HPP_

