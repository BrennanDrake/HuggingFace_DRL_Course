#!/bin/bash
# Simple monitoring script for CPU and GPU usage

echo "=== GPU Usage ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
awk -F', ' '{printf "GPU %s (%s):\n  GPU Util: %s%%\n  Memory Util: %s%% (%s/%s MB)\n  Temp: %s°C\n  Power: %s W\n\n", $1, $2, $3, $4, $5, $6, $7, $8}'

echo "=== CPU Usage ==="
read -r _ u1 n1 s1 i1 iw1 ir1 si1 st1 _ < /proc/stat
total1=$((u1 + n1 + s1 + i1 + iw1 + ir1 + si1 + st1))
idle1=$((i1 + iw1))
sleep 0.5
read -r _ u2 n2 s2 i2 iw2 ir2 si2 st2 _ < /proc/stat
total2=$((u2 + n2 + s2 + i2 + iw2 + ir2 + si2 + st2))
idle2=$((i2 + iw2))
cpu_used=$(awk -v t1="$total1" -v t2="$total2" -v i1="$idle1" -v i2="$idle2" 'BEGIN {dt=t2-t1; di=i2-i1; if (dt<=0) {printf "0.0"} else {printf "%.1f", (dt-di)*100/dt}}')
cpu_idle=$(awk -v t1="$total1" -v t2="$total2" -v i1="$idle1" -v i2="$idle2" 'BEGIN {dt=t2-t1; di=i2-i1; if (dt<=0) {printf "0.0"} else {printf "%.1f", di*100/dt}}')
echo "CPU Idle: ${cpu_idle}%"
echo "CPU Used: ${cpu_used}%"

echo "=== Process CPU Usage (Top 5) ==="
ps aux --sort=-%cpu | head -6 | awk '{printf "%-8s %6.1f%% %s\n", $11, $3, substr($0, index($0,$11))}'
