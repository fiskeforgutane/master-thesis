# The baseline used by all
baseline="add-random 0.01 ? remove-random 0.01 ? dedup"

while read mutation; do
	for p in 0.1
	do
		./target/release/master.exe --problem ~/master-playground/mirplib-rs/t45/LR2_22_DR3_333_VC4_V17a.json --log info --termination "1800 timeout 600 no-improvement |" rolling-horizon --population 3 --children 3 --tournament 2 --subproblem-size 60 --mutation "$baseline $mutation $p ?" 2>> log.txt
	done
done < mutations.txt
