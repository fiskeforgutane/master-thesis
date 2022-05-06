# git pull;
# rm -rf target/;
# sleep 2.0;

export set TIME="t60";

cat problems.txt | xargs -d ' ' -I {} cargo run --release -- ../master-playground/mirplib-rs/$TIME/{}