drop database salon;
create database salon;
--
alter database salon owner to freecodecamp;
--
\connect salon
--
create table public.customers (
    customer_id serial primary key,
    name character varying(30) not null,
    phone character varying(30) not null,
    constraint customer_phone_unique unique (phone)
);
create table public.services (
    service_id serial primary key,
    name character varying(30) not null
);
create table public.appointments (
    appointment_id serial primary key,
    customer_id integer not null,
    service_id integer not null,
    time character varying(30) not null,
    constraint fk_customer foreign key(customer_id) references customers(customer_id),
    constraint fk_service foreign key(service_id) references services(service_id)
);
--
alter table public.customers owner to freecodecamp;
alter table public.appointments owner to freecodecamp;
alter table public.services owner to freecodecamp;
--
insert into public.services values (nextval('services_service_id_seq'), 'service1');
insert into public.services values (nextval('services_service_id_seq'), 'service2');
insert into public.services values (nextval('services_service_id_seq'), 'service3');