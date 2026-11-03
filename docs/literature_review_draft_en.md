# Literature Review Draft — LSTM-Encoded Fuzzy Local Experts

**Status:** An English starting section for the final Jupyter Notebook, 30 October 2026. Verify the final bibliography format and add discussion of the team's actual results before submission.

## Method lineage and taxonomy

The proposed model combines two families of methods. An LSTM handles an ordered, multivariate history; a Takagi–Sugeno-style fuzzy layer combines simple local return models. The distinction matters because the LSTM's hidden features are learned from the entire 20-day sequence, while the fuzzy layer receives only a three-dimensional summary at a single forecast time.

1. **Local fuzzy models (1985).** Takagi and Sugeno described fuzzy antecedents whose consequents are local input–output models. This supplies the basic form used here: rule strength times local numerical prediction, followed by aggregation. Our two affine experts are a deliberately small version of that idea, with Gaussian membership functions over learned rather than hand-selected inputs. [Original paper](https://doi.org/10.1109/TSMC.1985.6313399).
2. **Trainable neuro-fuzzy inference (1993).** Jang's ANFIS placed fuzzy inference in an adaptive-network framework and showed how rule parameters could be learned from examples. It is relevant as the classical neuro-fuzzy alternative, but a direct ANFIS on all `20 days × 6 variables` would not express the intended sequential encoder and could make the rule space unnecessarily large. Our LSTM first reduces history to three learned inputs, and all parts are trained jointly by backpropagation. [Original paper](https://doi.org/10.1109/21.256541).
3. **Recurrent sequence representation (1997).** Hochreiter and Schmidhuber introduced LSTM to make learning longer temporal dependencies easier than in conventional recurrent networks. In this project the LSTM is an encoder, and an ordinary LSTM with a scalar return head is the directly relevant benchmark. The benchmark and hybrid receive identical information, so their difference isolates the value of the prediction head more clearly than comparing different datasets or look-back periods. [Original paper](https://doi.org/10.1162/neco.1997.9.8.1735).
4. **Deep sequence encoder plus fuzzy local forecasting (2025).** Ožbot, Škrjanc and Štruc's *Fuzzformer* combines an LSTM, multi-head temporal attention, learned low-dimensional representations, Gaussian fuzzy rules and ARIX local models for multi-horizon S&P 500 forecasting. Their study motivates letting a sequence model learn fuzzy-system inputs. Their paper also uses clustering and rule-separation objectives to support interpretable local models. The present course design removes attention, uses one next-day SRT ETF return, two diagonal-Gaussian rules, two affine experts, and one ordinary MSE objective. It should therefore be described as an **adaptation**, not a reproduction or a claim to obtain the paper's results. [Original paper](https://arxiv.org/html/2510.00960v1).

| Family | Uses time ordering? | Local fuzzy rules? | Role in this project |
| --- | --- | --- | --- |
| No-change return predictor | No learned sequence model | No | Noise-resistant reference for daily ETF returns |
| Ordinary LSTM | Yes | No | Main machine-learning benchmark |
| LSTM → 3D → linear head | Yes | No | Controls for representation compression |
| Classical ANFIS / Takagi–Sugeno-style models | Usually use a fixed-size input vector | Yes | Historical method family; explains rule/consequent form |
| Fuzzformer | Yes, with attention | Yes, with ARIX local models | Recent architecture that motivates the hybrid |
| Our LSTM → 3D → two fuzzy affine experts | Yes | Yes | Small course adaptation on SRT and published local market data |

## Why this adaptation is testable

The first hypothesis is about **forecasting**, not guaranteed superiority: `Fuzzy2` might lower the next-day raw-return RMSE relative to the same-input `LSTM32` or `LSTM3`. The `LSTM3` control is necessary because a better result could come from forcing a compact representation, independently of fuzzy rules. The second hypothesis is about **local specialization**: rule weights and expert outputs might vary meaningfully across observed past-volatility or past-trend conditions. If one rule dominates or both experts predict almost the same return, the intended specialization has not appeared, even if the forecast error is good.

Input-source ablations ask whether adding published SORA and STI improves the SRT forecast beyond SRT history alone. This is a data question, not evidence that either input causes ETF price movements. SORA is aligned by its publication time; the Monetary Authority of Singapore says the rate for a business day is published at 9:00 on the next business day. [MAS publication policy](https://www.mas.gov.sg/monetary-policy/SORA).

The Fuzzformer paper reports experiments on a different market, prediction horizon, target definition, normalization and architecture. Its numeric RMSE values cannot be placed in our result table as a direct head-to-head benchmark. We can compare **method design and observed strengths/limitations** in prose, while our own ordinary LSTM supplies the numerical benchmark on the identical SRT folds.

## Results paragraphs to complete after training

- State the pooled forward-validation and final-holdout return RMSE for `Z0`, `LSTM32`, `LSTM3` and `Fuzzy2`, including seed variation and paired differences on common target dates.
- Explain which data-source additions helped or hurt. If the gain is small compared with seed variation, say so.
- Discuss `K=1/2/3` rule sensitivity, the learned rule-weight distribution and whether expert outputs differ. Do not label a rule “bull” or “crash” based only on its index.
- Compare forecast accuracy with the separately timed, price-only next-open portfolio exercise. State the close-to-close forecast versus open-to-open trading horizon mismatch, cost assumptions and dividend limitation.
- End with limitations and future work: short ETF history, three learned latent features with no direct economic names, possible regime instability, missing one open, and the need to validate distributions before any total-return claim.

## Bibliography for the final Notebook

- Takagi, T., & Sugeno, M. (1985). *Fuzzy identification of systems and its applications to modeling and control*. IEEE Transactions on Systems, Man, and Cybernetics, SMC-15(1), 116–132. https://doi.org/10.1109/TSMC.1985.6313399
- Jang, J.-S. R. (1993). *ANFIS: Adaptive-network-based fuzzy inference system*. IEEE Transactions on Systems, Man, and Cybernetics, 23(3), 665–685. https://doi.org/10.1109/21.256541
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
- Ožbot, M., Škrjanc, I., & Štruc, V. (2025). *A Neuro-Fuzzy System for Interpretable Long-Term Stock Market Forecasting*. arXiv:2510.00960v1. https://arxiv.org/html/2510.00960v1
- Ang, K. K. (2026). *AI6124 — Week 8: Neural in Neuro-Fuzzy Systems*. Course lecture PDF supplied to the class.
