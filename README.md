# RAG-Scholar Minimal Example

Requires Python 3.12 and the packages in `requirements.txt`. CPU execution is supported. Three source PDFs, preprocessed MinerU materials, and nine queries in `data/queries.json` are included.

```shell
python -m pip install -r requirements.txt
python run_demo.py --download-model
```

The first run downloads BCE and CLIP, trains the GNN for 40 epochs, and runs all nine queries and image cropping. Generated files are written to `outputs/`; check `outputs/validation.json`. Model weights and generated outputs are not bundled.

With cached weights, use `python run_demo.py`. After a successful run, reuse the index with `python run_demo.py --reuse-index`. Local weights can be supplied with `--bce-model-path /path/to/bce` and `--clip-model-path /path/to/clip`. Use `--device cuda:0` for BCE/GNN on a compatible GPU; CLIP runs on CPU.

Run the tests independently:

```shell
python -B -m unittest discover -s tests -v
```

For optional answer generation, provide a local Qwen2-VL-7B-Instruct checkpoint:

```shell
python run_demo.py --reuse-index --answer-model-path /path/to/Qwen2-VL-7B-Instruct --device cuda:0
```

Answers are saved to `outputs/answers/`. Without this option, the demo runs retrieval and cropping only.
