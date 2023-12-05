echo Running DeepWalk algorithm with varied walk lengths
echo Window size: 2, Embedding size: 10 Walks per vertex: 10
echo ------------------------------------------------------
> karate_tune_walk_length_metrics.txt
echo Training....
for i in {10..400..10}; do 
    python3 deepwalk.py karate 2 10 10 $i >> karate_tune_walk_length_metrics.txt
done
echo Done!
echo ------------------------------------------------------
echo Plotting results...
python3 read_metrics.py karate_tune_walk_length_metrics.txt walklength
echo Done! See plots/karate_graph_walk_length.png for results.