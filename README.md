# Repository

Companion code for a Medium article.

## Business context

*Using linearmodels PanelOLS with region × COVID interaction terms to isolate the causal effect of the pandemic on US rail travel*

Amtrak ridership collapsed in 2020. That much is obvious from any chart. The harder question is whether the collapse was uniform — or whether COVID hit some regions harder than others, and whether the recovery followed the same pattern in the Northeast corridor as it did in the West.

Ordinary regression cannot answer this cleanly. Stations differ systematically: a Boston South Station handles orders of magnitude more passengers than a rural stop in Montana. If you ignore these fixed differences and pool the data, your COVID coefficient absorbs station-level heterogeneity and the estimate is biased.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).