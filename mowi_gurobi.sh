cd ~/master-thesis && cargo build --release;

export i=0;
export pids="";
for file in $(find ~/master-playground/mowi-rs/toy/*/*/*.json)
do
    echo "i=$i: Starting run on $file"
    ~/master-thesis/target/release/master --problem $file --log info --termination "7200 timeout" exact >> log-$(hostname)-$i.txt
done