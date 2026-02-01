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
  static void ToProbabilities(std::vector<double>* p_rec) {
    std::vector<double> &rec = *p_rec;
    double wmax = rec[0];
    for (size_t i = 1; i < rec.size(); ++i) {
      wmax = std::max(rec[i], wmax);
    }
    double wsum = 0.0f;
    for (size_t i = 0; i < rec.size(); ++i) {
      rec[i] = std::pow(2, rec[i] - wmax);
      wsum += rec[i];
    }
    for (size_t i = 0; i < rec.size(); ++i) {
      rec[i] /= static_cast<double>(wsum);
    }
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

    // Resize phi_ to 2D: num_data_ x num_class_
    phi_.resize(num_data_);
    for (int i = 0; i < num_data_; ++i) {
      int encoded_value = static_cast<int>(label_[i]);
      std::vector<int> label_int_2d_i = MixedRadixDecode(encoded_value, num_class_);
      phi_[i].resize(num_class_);
      for (int k = 0; k < num_class_; ++k) {
        phi_[i][k] = num_class_ - static_cast<int>(label_int_2d_i[k]);
      }
      ToProbabilities(&phi_[i]);
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
          std::vector<double> rec(num_pred_per_row);
          objective->ConvertOutput(raw_score.data(), rec.data());
          // add loss
          sum_loss += LossOnPoint(phi_[i], &rec);
        }
      } else {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          std::vector<double> raw_score(num_tree_per_iteration);
          for (int k = 0; k < num_tree_per_iteration; ++k) {
            size_t idx = static_cast<size_t>(num_data_) * k + i;
            raw_score[k] = static_cast<double>(score[idx]);
          }
          std::vector<double> rec(num_pred_per_row);
          objective->ConvertOutput(raw_score.data(), rec.data());
          // add loss
          sum_loss += LossOnPoint(phi_[i], &rec) * weights_[i];
        }
      }
    } else {
      if (weights_ == nullptr) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          std::vector<double> rec(num_tree_per_iteration);
          for (int k = 0; k < num_tree_per_iteration; ++k) {
            size_t idx = static_cast<size_t>(num_data_) * k + i;
            rec[k] = static_cast<double>(score[idx]);
          }
          // add loss
          sum_loss += LossOnPoint(phi_[i], &rec);
        }
      } else {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          std::vector<double> rec(num_tree_per_iteration);
          for (int k = 0; k < num_tree_per_iteration; ++k) {
            size_t idx = static_cast<size_t>(num_data_) * k + i;
            rec[k] = static_cast<double>(score[idx]);
          }
          // add loss
          sum_loss += LossOnPoint(phi_[i], &rec) * weights_[i];
        }
      }
    }
    double loss = sum_loss / sum_weights_;
    return std::vector<double>(1, loss);
  }

  inline static double LossOnPoint(const std::vector<double>& phi, std::vector<double>* score) {
    // Compute cross-entropy loss: -sum_k(phi_k * log(rho_k))
    double loss = 0.0;
    auto& rho = *score;
    for (size_t k = 0; k < phi.size(); ++k) {
      double rho_k = std::max<double>(rho[k], kEpsilon);
      loss -= phi[k] * std::log(rho_k);
    }
    return loss;
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
  /*! \brief Pre-computed phi (ideal distribution) for each data point */
  std::vector<std::vector<double>> phi_;
};

/*! \brief Soft Cross-Entropy metric for multiclass task with probability labels */
class SoftCrossEntropyMetric: public Metric {
 public:
  explicit SoftCrossEntropyMetric(const Config& config) : config_(config) {
    num_class_ = config.num_class;
  }

  virtual ~SoftCrossEntropyMetric() {}

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return -1.0f;
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    name_.emplace_back("soft_multiclass");
    num_data_ = num_data;
    // get label probabilities from init_score
    label_probs_ = metadata.init_score();
    // get weights
    weights_ = metadata.weights();
    if (weights_ == nullptr) {
      sum_weights_ = static_cast<double>(num_data_);
    } else {
      sum_weights_ = 0.0;
      for (data_size_t i = 0; i < num_data_; ++i) {
        sum_weights_ += weights_[i];
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
          std::vector<double> label_probs(num_tree_per_iteration);
          for (int k = 0; k < num_tree_per_iteration; ++k) {
            size_t idx = static_cast<size_t>(num_data_) * k + i;
            raw_score[k] = static_cast<double>(score[idx]);
            label_probs[k] = label_probs_[idx];
          }
          std::vector<double> rec(num_pred_per_row);
          objective->ConvertOutput(raw_score.data(), rec.data());
          // add loss
          sum_loss += LossOnPoint(&label_probs, &rec);
        }
      } else {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          std::vector<double> raw_score(num_tree_per_iteration);
          std::vector<double> label_probs(num_tree_per_iteration);
          for (int k = 0; k < num_tree_per_iteration; ++k) {
            size_t idx = static_cast<size_t>(num_data_) * k + i;
            raw_score[k] = static_cast<double>(score[idx]);
            label_probs[k] = label_probs_[idx];
          }
          std::vector<double> rec(num_pred_per_row);
          objective->ConvertOutput(raw_score.data(), rec.data());
          // add loss
          sum_loss += LossOnPoint(&label_probs, &rec) * weights_[i];
        }
      }
    } else {
      if (weights_ == nullptr) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          std::vector<double> rec(num_tree_per_iteration);
          std::vector<double> label_probs(num_tree_per_iteration);
          for (int k = 0; k < num_tree_per_iteration; ++k) {
            size_t idx = static_cast<size_t>(num_data_) * k + i;
            rec[k] = static_cast<double>(score[idx]);
            label_probs[k] = label_probs_[idx];
          }
          // add loss
          sum_loss += LossOnPoint(&label_probs, &rec);
        }
      } else {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          std::vector<double> rec(num_tree_per_iteration);
          std::vector<double> label_probs(num_tree_per_iteration);
          for (int k = 0; k < num_tree_per_iteration; ++k) {
            size_t idx = static_cast<size_t>(num_data_) * k + i;
            rec[k] = static_cast<double>(score[idx]);
            label_probs[k] = label_probs_[idx];
          }
          // add loss
          sum_loss += LossOnPoint(&label_probs, &rec) * weights_[i];
        }
      }
    }
    double loss = sum_loss / sum_weights_;
    return std::vector<double>(1, loss);
  }

  inline static double LossOnPoint(const std::vector<double>* label_probs, const std::vector<double>* score) {
    // Compute cross-entropy loss: -sum_k(y_k * log(p_k))
    double loss = 0.0;
    const auto& y = *label_probs;
    const auto& p = *score;
    for (size_t k = 0; k < y.size(); ++k) {
      double p_k = std::max<double>(p[k], kEpsilon);
      loss -= y[k] * std::log(p_k);
    }
    return loss;
  }

 private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer to label probabilities from init_score */
  const double* label_probs_;
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
};

}  // namespace LightGBM
#endif  // LIGHTGBM_SRC_METRIC_CUSTOM_METRIC_HPP_

