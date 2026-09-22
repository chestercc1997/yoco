# Source notice

The YOCO decoder implementation under `runtime/yoco_official/` is derived from
Microsoft's UniLM YOCO release at commit
`ca43e4cd19445a536f133bf2bc25b573b2f0c7c5`, licensed under MIT. The provenance
manifest and the small V100 compatibility patch are stored in `provenance/`.

The paired Transformer, training loop, evaluation adapter, benchmarks, plots,
and experiment organization were prepared for this reproduction. The public
training corpus is FineWeb-Edu sample-10BT, rather than the original private
YOCO training mixture.
