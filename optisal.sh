cd ~/master-thesis && cargo build --release;

export i=0;
export pids="";
for file in $(find ~/master-playground/mirplib-rs/t45/*.json)
do
    echo "Starting run on $file"
    ~/master-thesis/target/release/master --problem file --log info --termination "7200 timeout 1800 no-improvement no-violation full-empty-valid & & |" rolling-horizon --threads 1 --full-penalty-after 3600000 --population 3 --children 3 --tournament 2 --subproblem-size 45 --mutation "std" 2>> log-$(hostname)-$i.txt &
    pids="$pids $!"
    i=$((i + 1))

    if [ $i -eq $1 ]; then
        wait "$pids"
        i=$((0))
        pids=""
    fi
done