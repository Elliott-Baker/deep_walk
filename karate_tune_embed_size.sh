echo Running DeepWalk algorithm with varied embedding sizes
echo Window size: 2, Walks per vertex: 2, Walk length: 100
echo ------------------------------------------------------
> karate_tune_embed_size_metrics.txt
echo Training....
for i in {10..1000..10}; do 
    python3 deepwalk.py karate 2 $i 2 100 >> karate_tune_embed_size_metrics.txt
done
echo Done!
echo ------------------------------------------------------
echo Plotting results...
python3 read_metrics.py karate_tune_embed_size_metrics.txt embeddingsize
echo Done! See plots/karate_graph_embed_size.png for results.

