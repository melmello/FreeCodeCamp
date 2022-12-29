#!/bin/bash

PSQL="psql -X --username=freecodecamp --dbname=number_guess --tuples-only --no-align -c"

echo "Enter your username:"
read username

name="$($PSQL "SELECT username FROM users WHERE username='$username'")"
games_played="$($PSQL "SELECT count(*) FROM users INNER JOIN games using(user_id) WHERE username= '$name'")"
best_game="$($PSQL "SELECT min(best_guess) FROM users INNER JOIN games using(user_id) WHERE username= '$name'")"

if [[ -z $name ]]
then
INSERT_USERNAME="$($PSQL "INSERT INTO users(username) values('$username')")"
echo "Welcome, $username! It looks like this is your first time here."
else
echo "Welcome back, $name! You have played $games_played games, and your best game took $best_game guesses."
fi

random=$(( RANDOM % 1000 + 1 ))
guess=1
echo "Guess the secret number between 1 and 1000:"
while read input_number
do
  if [[ ! $input_number =~ ^[0-9]+$ ]]
    then
    echo "That is not an integer, guess again:"
    else
      if [[ $input_number -eq $random ]]
      then
      break;
      else
        if [[ $input_number -gt $random ]]
        then
         echo "It's lower than that, guess again:"
        elif [[ $input_number -lt $random ]]
        then
         echo "It's higher than that, guess again:"
        fi
      fi
  fi
  guess=$(( $guess + 1 ))
done

 if [[ $guess -eq 1 ]]
  then
   echo "You guessed it in $guess tries. The secret number was $random. Nice job!"
  else
   echo "You guessed it in $guess tries. The secret number was $random. Nice job!"
 fi

user_id="$($PSQL "SELECT user_id FROM users WHERE username='$username'")"
INSERT_GAME="$($PSQL "INSERT INTO games(best_guess,user_id) values($guess,$user_id)")"