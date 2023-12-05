echo Running DeepWalk algorithm with varied walk lengths
echo Window size: 2, Embedding size: 10 Walks length: 200
echo ------------------------------------------------------
> karate_tune_walks_per.txt
echo Training....
for i in {2..25..2}; do 
    python3 deepwalk.py karate 2 10 $i 200 >> karate_tune_walks_per.txt
done
echo Done!
echo ------------------------------------------------------
echo Plotting results...
python3 read_metrics.py karate_tune_walks_per.txt
echo Done! See plots/karate_graph_metrics2.png for results.