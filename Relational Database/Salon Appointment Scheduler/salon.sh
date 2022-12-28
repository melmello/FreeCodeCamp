#!/bin/bash

PSQL="psql -X --username=freecodecamp --dbname=salon --tuples-only -c"

echo -e "\n\tWelcome\n"


MAIN_MENU(){

SERVICES="$($PSQL "select * from services order by service_id")"

echo -e "\nPlease select a service\n"

echo "$SERVICES" | while read SERVICE_ID BAR SERVICE
do
echo "$SERVICE_ID) $SERVICE"
done
SERVICES
}
SERVICES() {

read SERVICE_ID_SELECTED

SERVICE_AVAILABILITY=$($PSQL "select service_id from services where service_id=$SERVICE_ID_SELECTED")
if [[ -z $SERVICE_AVAILABILITY ]]
then
echo -e "\nService not found\n"

echo "$SERVICES" | while read SERVICE_ID BAR SERVICE
do
echo "$SERVICE_ID) $SERVICE"
done
SERVICES
else
echo -e "\nPlease enter the phone number\n"
read CUSTOMER_PHONE

CUSTOMER_ID=$($PSQL "select customer_id from customers where phone='$CUSTOMER_PHONE'")
if [[ -z $CUSTOMER_ID ]]
then
echo -e "\nPlease enter your name\n"
read CUSTOMER_NAME
INSERT_CUSTOMER_RESULT=$($PSQL "insert into customers(phone,name) values('$CUSTOMER_PHONE','$CUSTOMER_NAME')")

SERVICE_NAME=$($PSQL "select name from services where service_id=$SERVICE_ID_SELECTED")
echo -e "\nPlease enter the time for $SERVICE_NAME, $CUSTOMER_NAME?\n"
read SERVICE_TIME
CUSTOMER_ID=$($PSQL "select customer_id from customers where phone='$CUSTOMER_PHONE'")

INSERT_SERVICE_RESULT=$($PSQL "INSERT INTO appointments(time,customer_id,service_id) values('$SERVICE_TIME',$CUSTOMER_ID,$SERVICE_ID_SELECTED)")
echo -e "\nI have put you down for a $SERVICE_NAME at $SERVICE_TIME, $CUSTOMER_NAME.\n"

else
SERVICE_NAME=$($PSQL "select name from services where service_id=$SERVICE_ID_SELECTED")
CUSTOMER_NAME=$($PSQL "select name from customers where phone='$CUSTOMER_PHONE'")
echo -e "\nPlease enter the time for $SERVICE_NAME, $CUSTOMER_NAME?\n"
read SERVICE_TIME

INSERT_SERVICE_RESULT=$($PSQL "insert into appointments(time,customer_id,service_id) values('$SERVICE_TIME',$CUSTOMER_ID,$SERVICE_ID_SELECTED)")
echo -e "\nI have put you down for a $SERVICE_NAME at $SERVICE_TIME, $CUSTOMER_NAME.\n"

fi
fi
}

MAIN_MENU