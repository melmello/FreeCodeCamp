#!/bin/bash

PSQL="psql -X --username=freecodecamp --dbname=periodic_table  --no-align --tuples-only -c"

if [[ $1 ]]
then
    if [[ $1 =~ ^[0-9]+$ ]]
    then
        atomic_number=$1
        name="$($PSQL "select name from elements where atomic_number = $atomic_number")"
        if [[ -z $name ]]
        then
            echo "I could not find that element in the database."
        else
            symbol="$($PSQL "select symbol from elements where atomic_number = $atomic_number")"
            atomic_mass="$($PSQL "select atomic_mass from properties where atomic_number = $atomic_number")"
            melting_point_celsius="$($PSQL "select melting_point_celsius from properties where atomic_number = $atomic_number")"
            boiling_point_celsius="$($PSQL "select boiling_point_celsius from properties where atomic_number = $atomic_number")"
            type="$($PSQL "select type from properties full join types using(type_id) where atomic_number = $atomic_number")"
            echo The element with atomic number $atomic_number is $name \("$symbol"\). It\'s a $type, with a mass of $atomic_mass amu. $name has a melting point of $melting_point_celsius celsius and a boiling point of $boiling_point_celsius celsius.
        fi
    elif (( `echo -n "$1" | wc -m ` <= 2 )) #`echo -n "$str"|wc -c`   the echo command is given an argument -n which does escape the new line character.
    then
        symbol=$1
        name="$($PSQL "select name from elements where symbol = '$symbol'")"
        if [[ -z $name ]]
        then
            echo "I could not find that element in the database."
        else
            atomic_number="$($PSQL "select atomic_number from elements where symbol = '$symbol'")"
            atomic_mass="$($PSQL "select atomic_mass from properties where atomic_number = $atomic_number")"
            melting_point_celsius="$($PSQL "select melting_point_celsius from properties where atomic_number = $atomic_number")"
            boiling_point_celsius="$($PSQL "select boiling_point_celsius from properties where atomic_number = $atomic_number")"
            type="$($PSQL "select type from properties full join types using(type_id) where atomic_number = $atomic_number")"
            echo The element with atomic number $atomic_number is $name \("$symbol"\). It\'s a $type, with a mass of $atomic_mass amu. $name has a melting point of $melting_point_celsius celsius and a boiling point of $boiling_point_celsius celsius.
        fi
    else
        name=$1
        
        symbol="$($PSQL "select symbol from elements where name = '$name'")"
        if [[ -z $symbol ]]
        then
            echo "I could not find that element in the database."
        else
            atomic_number="$($PSQL "select atomic_number from elements where symbol = '$symbol'")"
            atomic_mass="$($PSQL "select atomic_mass from properties where atomic_number = $atomic_number")"
            melting_point_celsius="$($PSQL "select melting_point_celsius from properties where atomic_number = $atomic_number")"
            boiling_point_celsius="$($PSQL "select boiling_point_celsius from properties where atomic_number = $atomic_number")"
            type="$($PSQL "select type from properties full join types using(type_id) where atomic_number = $atomic_number")"
            echo The element with atomic number $atomic_number is $name \("$symbol"\). It\'s a $type, with a mass of $atomic_mass amu. $name has a melting point of $melting_point_celsius celsius and a boiling point of $boiling_point_celsius celsius.
        fi
    fi
else
    echo "Please provide an element as an argument."
fi