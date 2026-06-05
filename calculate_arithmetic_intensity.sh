# arithmetic_intensity.py B H N d median_ms BLOCK_SIZE_N
echo "--------------------------------------------------"
python3 arithmetic_intensity.py 2 32 1024 4096 0.490 32
echo "--------------------------------------------------"
python3 arithmetic_intensity.py 2 32 2048 4096 1.829 32
echo "--------------------------------------------------"
python3 arithmetic_intensity.py 2 32 4096 4096 7.177 32
echo "--------------------------------------------------"
python3 arithmetic_intensity.py 2 32 1024 4096 0.582 64
echo "--------------------------------------------------"
python3 arithmetic_intensity.py 2 32 2048 4096 2.159 64
echo "--------------------------------------------------"
python3 arithmetic_intensity.py 2 32 4096 4096 8.481 64
echo "--------------------------------------------------"
python3 arithmetic_intensity.py 2 32 1024 4096 10.843 128
echo "--------------------------------------------------"
python3 arithmetic_intensity.py 2 32 2048 4096 42.106 128
echo "--------------------------------------------------"
python3 arithmetic_intensity.py 2 32 4096 4096 166.648 128
echo "--------------------------------------------------"