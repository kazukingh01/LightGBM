/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_OBJECTIVE_CUSTOM_OBJECTIVE_HPP_
#define LIGHTGBM_SRC_OBJECTIVE_CUSTOM_OBJECTIVE_HPP_

#include "multiclass_objective.hpp"

namespace LightGBM {
/*!
 * \brief Focal Loss objective function for multiclass classification
 */
class FocalLossSoftmax: public MulticlassSoftmax {
 public:
  explicit FocalLossSoftmax(const Config& config) : MulticlassSoftmax(config) {
    gamma_ = config.focal_gamma;
  }

  explicit FocalLossSoftmax(const std::vector<std::string>& strs) : MulticlassSoftmax(strs) {
    gamma_ = 1.0;  // default value for model loading
    for (auto str : strs) {
      auto tokens = Common::Split(str.c_str(), ':');
      if (tokens.size() == 2) {
        if (tokens[0] == std::string("focal_gamma")) {
          Common::Atof(tokens[1].c_str(), &gamma_);
        }
      }
    }
  }

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    if (weights_ == nullptr) {
      std::vector<double> rec;
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) private(rec)
      for (data_size_t i = 0; i < num_data_; ++i) {
        rec.resize(num_class_);
        for (int k = 0; k < num_class_; ++k) {
          size_t idx = static_cast<size_t>(num_data_) * k + i;
          rec[k] = static_cast<double>(score[idx]);
        }
        Common::Softmax(&rec);
        double Z = 0.0;
        double D = 0.0;
        double p_label = 0.0;
        for (int k = 0; k < num_class_; ++k) {
          if (label_int_[i] == k) {
            auto p = rec[k];
            auto l = std::log(p);
            auto A = gamma_ * p * l + p - 1.0;
            auto B = std::pow(1.0 - p, gamma_ - 1.0);
            p_label = p;
            Z = A * B;
            D = gamma_ / (1.0 - p) * B * (-1.0 * A + l - p + 1.0);
          }
        }
        for (int k = 0; k < num_class_; ++k) {
          auto p = rec[k];
          size_t idx = static_cast<size_t>(num_data_) * k + i;
          if (label_int_[i] == k) {
            gradients[idx] = static_cast<score_t>((1.0 - p) * Z);
            hessians[idx] = static_cast<score_t>(factor_ * (-1.0 * Z * p * (1.0 - p) + p * (1.0 - p) * (1.0 - p) * D));
          } else {
            gradients[idx] = static_cast<score_t>(-1.0 * p * Z);
            hessians[idx] = static_cast<score_t>(factor_ * (-1.0 * Z * p * (1.0 - p) + p * p * p_label * D));
          }
        }
      }
    } else {
      std::vector<double> rec;
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) private(rec)
      for (data_size_t i = 0; i < num_data_; ++i) {
        rec.resize(num_class_);
        for (int k = 0; k < num_class_; ++k) {
          size_t idx = static_cast<size_t>(num_data_) * k + i;
          rec[k] = static_cast<double>(score[idx]);
        }
        Common::Softmax(&rec);
        double Z = 0.0;
        double D = 0.0;
        double p_label = 0.0;
        for (int k = 0; k < num_class_; ++k) {
          if (label_int_[i] == k) {
            auto p = rec[k];
            auto l = std::log(p);
            auto A = gamma_ * p * l + p - 1.0;
            auto B = std::pow(1.0 - p, gamma_ - 1.0);
            p_label = p;
            Z = A * B;
            D = gamma_ / (1.0 - p) * B * (-1.0 * A + l - p + 1.0);
          }
        }
        for (int k = 0; k < num_class_; ++k) {
          auto p = rec[k];
          size_t idx = static_cast<size_t>(num_data_) * k + i;
          if (label_int_[i] == k) {
            gradients[idx] = static_cast<score_t>((1.0 - p) * Z * weights_[i]);
            hessians[idx] = static_cast<score_t>(factor_ * (-1.0 * Z * p * (1.0 - p) + p * (1.0 - p) * (1.0 - p) * D) * weights_[i]);
          } else {
            gradients[idx] = static_cast<score_t>(-1.0 * p * Z * weights_[i]);
            hessians[idx] = static_cast<score_t>(factor_ * (-1.0 * Z * p * (1.0 - p) + p * p * p_label * D) * weights_[i]);
          }
        }
      }
    }
  }

  const char* GetName() const override {
    return "focalloss";
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName() << " ";
    str_buf << "num_class:" << num_class_ << " ";
    str_buf << "focal_gamma:" << gamma_;
    return str_buf.str();
  }

 private:
  double gamma_;
};


/*!
* \brief Objective function for multiclass classification with XE-NDCG style gradients
*/
class XendcgSoftmax: public MulticlassSoftmax {
 public:
  explicit XendcgSoftmax(const Config& config) : MulticlassSoftmax(config) {
  }

  explicit XendcgSoftmax(const std::vector<std::string>& strs) : MulticlassSoftmax(strs) {
  }

  /*!
   * \brief Decode a mixed-radix encoded value into a vector of digits
   * \param value The encoded integer value
   * \param radices The radix for each position (all num_class_ in this case)
   * \return Vector of decoded digits
   */
  static std::vector<int> MixedRadixDecode(int value, const std::vector<int>& radices) {
    std::vector<int> digits;
    digits.reserve(radices.size());
    for (int radix : radices) {
      digits.push_back(value % radix);
      value /= radix;
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
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();

    // Build radices: [num_class_, num_class_, ..., num_class_] (num_class_ elements)
    std::vector<int> radices(num_class_, num_class_);

    // Resize label_int_2d_ to 2D: num_data_ x num_class_
    phi_.resize(num_data_);
    for (int i = 0; i < num_data_; ++i) {
      int encoded_value = static_cast<int>(label_[i]);
      std::vector<int> label_int_2d_i = MixedRadixDecode(encoded_value, radices);
      phi_[i].resize(num_class_);
      for (int k = 0; k < num_class_; ++k) {
        phi_[i][k] = num_class_ - static_cast<int>(label_int_2d_i[k]);
      }
      ToProbabilities(&phi_[i]);
    }

    // Debug code
    // for (int i = 0; i < num_data_; ++i) {
    //   // Debug print phi_[i]
    //   std::string label_str = "phi_[" + std::to_string(i) + "] = [";
    //   for (size_t j = 0; j < phi_[i].size(); ++j) {
    //     label_str += std::to_string(phi_[i][j]);
    //     if (j < phi_[i].size() - 1) {
    //       label_str += ", ";
    //     }
    //   }
    //   label_str += "]";
    //   Log::Info(label_str.c_str());
    // }

    // Calculate (weighted) mean of each column
    class_init_probs_.resize(num_class_, 0.0);
    double sum_weight = 0.0;

    if (Network::num_machines() > 1) {
      // For distributed learning, compute global weighted mean
      sum_weight = Network::GlobalSyncUpBySum(sum_weight);
      for (int k = 0; k < num_class_; ++k) {
        class_init_probs_[k] = Network::GlobalSyncUpBySum(class_init_probs_[k]);
      }
    }

    for (int k = 0; k < num_class_; ++k) {
      class_init_probs_[k] /= sum_weight;
    }

  }

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    if (weights_ == nullptr) {
      std::vector<double> rec;
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) private(rec)
      for (data_size_t i = 0; i < num_data_; ++i) {
        rec.resize(num_class_);
        for (int k = 0; k < num_class_; ++k) {
          size_t idx = static_cast<size_t>(num_data_) * k + i;
          rec[k] = static_cast<double>(score[idx]);
        }
        Common::Softmax(&rec);
        for (int k = 0; k < num_class_; ++k) {
          auto rhok = rec[k];
          auto phik = phi_[i][k];
          size_t idx = static_cast<size_t>(num_data_) * k + i;
          gradients[idx] = static_cast<score_t>(-1.0f * phik + rhok);
          hessians[idx] = static_cast<score_t>(factor_ * rhok * (1.0f - rhok));
        }
      }
    } else {
      std::vector<double> rec;
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) private(rec)
      for (data_size_t i = 0; i < num_data_; ++i) {
        rec.resize(num_class_);
        for (int k = 0; k < num_class_; ++k) {
          size_t idx = static_cast<size_t>(num_data_) * k + i;
          rec[k] = static_cast<double>(score[idx]);
        }
        Common::Softmax(&rec);
        for (int k = 0; k < num_class_; ++k) {
          auto rhok = rec[k];
          auto phik = phi_[i][k];
          size_t idx = static_cast<size_t>(num_data_) * k + i;
          gradients[idx] = static_cast<score_t>((-1.0f * phik + rhok) * weights_[i]);
          hessians[idx] = static_cast<score_t>(factor_ * rhok * (1.0f - rhok) * weights_[i]);
        }
      }
    }
  }

  const char* GetName() const override {
    return "multirank";
  }

 private:
  /*! \brief Pre-computed phi values: 2D array of shape (num_data_, num_class_) */
  std::vector<std::vector<double>> phi_;
};


}  // namespace LightGBM
#endif  // LIGHTGBM_SRC_OBJECTIVE_CUSTOM_OBJECTIVE_HPP_



