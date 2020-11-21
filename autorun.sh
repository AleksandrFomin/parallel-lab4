#!/bin/bash

binary=$1
N1=$2
N2=$3
delta=$(((N2-N1)/10))

for i in {0..10}
do
	./${binary} $((N1+$i*$delta))
done