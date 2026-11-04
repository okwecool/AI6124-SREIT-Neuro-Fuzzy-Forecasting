# Literature Review and Result Comparison — LSTM-Encoded Fuzzy Local Experts

**Status:** Updated after the frozen 4 November 2026 final evaluation. This section is included in the single course Notebook.

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

## Observed result versus recent research

The cited Fuzzformer study uses S&P 500 **multi-horizon price forecasting** with an LSTM, temporal multi-head attention, low-dimensional latent spaces, 16 Gaussian rules and ARIX local models. Its training includes a winner-takes-all local objective and clustering, overlap and balance terms. Its reported test RMSE is broadly comparable to its ARIMA references and lower than its tested LSTM configurations in the reported horizons. The authors also show one decomposed forecast. These are the **paper's observations on its own task**; they do not imply that our two-rule adaptation will reproduce them. [Fuzzformer methods and experiments](https://arxiv.org/html/2510.00960v1).

Our study removes attention and ARIX, fixes two diagonal-Gaussian rules with simple centers, and trains all components using one global MSE objective on **one-day SRT ETF raw returns**. Development RMSE was 0.009489 for `Fuzzy2`, above the 0.009431 unchanged-price reference and the 0.009473 compressed `LSTM3` control. On the one frozen final period, `Fuzzy2` reached RMSE 0.006349 versus 0.006384 for ordinary `LSTM32` and 0.006377 for `Z0`. Its final MAE was worse than `Z0`, and non-Gaussian `Gate2` was close at RMSE 0.006353. These differences are small and metric-dependent; no cross-dataset numerical ranking against Fuzzformer is valid. Source-group differences were similarly small, and the seed-11 `K=3` diagnostic worsened development RMSE.

The mechanism result is more important than the small error ordering. Final `Fuzzy2` maximum rule weights average only about 0.51–0.56, with no day above 0.8; one rule has no test case above its training-defined 75th-percentile activation cut in one seed. The weights and affine contributions explain the model's arithmetic, but the data do not give them stable economic labels. The simplified architecture and global loss may help explain the weak specialization, although this is a **hypothesis**, not a causal attribution established by our ablations. A closer test would preregister a local expert objective or rule-separation term and evaluate it on new periods without replacing this frozen result.

The trading application further distinguishes predictive fit from decision value. The untuned `Fuzzy2` policy ends at 0.999 of initial wealth after 10-bps single-side costs, versus 1.000 for cash, although it exceeds losing always-long and momentum controls in this down test period. Two of three seeds never trade; the model's near-zero, mostly negative predictions explain much of this pattern. This does not establish profitable timing. The primary target is close-to-close while execution is next-open to following-open, and one open quote is unavailable. Price-only and issuer-checked gross-distribution ledgers are reported separately.

## Application areas and theoretical questions

Sequence-to-fuzzy models are plausible when ordered inputs interact with a small number of local response patterns: financial returns, energy demand, industrial process control and other sensor streams. Their transparency should be tested with activation coverage, locally matched errors, stable rule identities and decision outcomes rather than inferred from the presence of fuzzy terminology alone.

Three research questions follow from this experiment. **First**, what local loss or separation constraint improves expert specialization without increasing forecast error or creating empty rules? **Second**, can a learned latent space preserve rule meaning across seeds and rolling refits, or do centers need anchored observable variables? **Third**, how should a rule system adapt to concept drift while keeping its explanations stable and auditable? These questions require new preregistered periods or instruments; the present final holdout cannot be reused for model selection.

A non-financial follow-up could predict next-day building electricity demand from 20 days of load, temperature and occupancy schedules. The LSTM would encode the sequence; a small fuzzy head could represent differing demand responses during occupied and unoccupied conditions. Compare it with a same-input LSTM, compressed latent head and non-Gaussian mixture on chronological future weeks. Test both forecast error and whether high-activation rules consistently reduce local error in independently measured occupancy regimes. This would check semantic usefulness more directly than merely plotting membership weights.

## Bibliography for the final Notebook

- Takagi, T., & Sugeno, M. (1985). *Fuzzy identification of systems and its applications to modeling and control*. IEEE Transactions on Systems, Man, and Cybernetics, SMC-15(1), 116–132. https://doi.org/10.1109/TSMC.1985.6313399
- Jang, J.-S. R. (1993). *ANFIS: Adaptive-network-based fuzzy inference system*. IEEE Transactions on Systems, Man, and Cybernetics, 23(3), 665–685. https://doi.org/10.1109/21.256541
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
- Ožbot, M., Škrjanc, I., & Štruc, V. (2025). *A Neuro-Fuzzy System for Interpretable Long-Term Stock Market Forecasting*. arXiv:2510.00960v1. https://arxiv.org/html/2510.00960v1
- Ang, K. K. (2026). *AI6124 — Week 8: Neural in Neuro-Fuzzy Systems*. Course lecture PDF supplied to the class.
