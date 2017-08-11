#!/bin/bash

DIR=$1

for filename in `ls $DIR/*.mid`
do
	midi2abc "$filename" -o "$filename.abc"
done
