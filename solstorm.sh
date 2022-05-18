find ~/mirplib/$1/*.json | xargs -I file ./master --problem file --config config37.json --stuck-timeout 7200 all --write
