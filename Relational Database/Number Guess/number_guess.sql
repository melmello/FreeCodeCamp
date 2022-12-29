drop database number_guess;
create database number_guess;
--
alter database number_guess owner to freecodecamp;
--
\connect number_guess
--
create table public.users (
    user_id serial primary key,
    username character varying(22) not null
);
--
create table public.games (
    game_id serial primary key,
    best_guess integer not null,
    user_id integer not null,
    constraint fk_user foreign key(user_id) references users(user_id)
);
--
alter table public.users owner to freecodecamp;
alter table public.games owner to freecodecamp;