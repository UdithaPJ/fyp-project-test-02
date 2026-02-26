.venv\Scripts\activate

python scripts/02_wideparquet_to_memmap.py data/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_reads.parquet out
###
nice -n 15 ionice -c2 -n7 python scripts/02_wideparquet_to_memmap_fast2.py data/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_exon_reads.parquet out

(Start-Process -FilePath "python" -ArgumentList "scripts/02b_normalize_memmap.py out" -PassThru -NoNewWindow).PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
##
nice -n 10 python scripts/02b_normalize_memmap.py out

(Start-Process -FilePath "python" -ArgumentList "scripts/03_build_feature_topk_graph.py out 50" -PassThru -NoNewWindow).PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

(Start-Process -FilePath "python" -ArgumentList "scripts/04_cpu_baselines_feature_graph.py out 50" -PassThru -NoNewWindow).PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

.venv\Scripts\activate
pip install threadpoolctl

$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"

python scripts/03b_build_feature_topk_graph_parallel.py out 50 1 128 256
python scripts/03b_build_feature_topk_graph_parallel.py out 50 4 128 256

python scripts/04b_algos_parallel_compare.py out/feature_graph_top50_csr_parallel_w4.npz single
python scripts/04b_algos_parallel_compare.py out/feature_graph_top50_csr_parallel_w4.npz parallel

python scripts/05_compare_results.py out

# 0) activate venv

.venv\Scripts\activate

# 1) install gpu deps

pip install -U cupy-cuda11x threadpoolctl

# 2) run GPU algorithms on the existing CSR graph

python scripts/06_gpu_algos.py out/feature_graph_top50_csr_parallel_w4.npz out/gpu_results

# 3) compare CPU vs GPU (same graph)

python scripts/07_compare_cpu_gpu.py `  out/feature_graph_top50_csr_parallel_w4.npz`
out/gpu_results/gpu_results.json `
out/compare_cpu_gpu.json

###
python scripts/10_timing_summary.py --build_w1 out/graph_build_parallel_w1.json --build_wN out/graph_build_parallel_w4.json --cpu_single out/algo_results/single/summary.json --cpu_parallel out/algo_results/parallel/summary.json --gpu out/gpu_results/gpu_results.json --out_json out/timing_summary_combined.json
###
python scripts/10_timing_summary_combined.py --build_w1 out/graph_build_parallel_w1.json --build_wN out/graph_build_parallel_w4.json --cpu_single out/algo_results/single/summary.json --cpu_parallel out/algo_results/parallel/summary.json --gpu out/gpu_results/gpu_results.json
###