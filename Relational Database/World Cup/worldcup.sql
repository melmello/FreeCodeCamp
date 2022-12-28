drop database worldcup;
create database worldcup;
--
alter database worldcup owner to freecodecamp;
--
\connect worldcup
--
create table public.teams (
    team_id serial primary key,
    name character varying(30) not null,
    constraint team_name_unique unique (name)
);
create table public.games (
    game_id serial primary key,
    winner_id integer not null,
    opponent_id integer not null,
    year integer not null,
    round character varying(30) not null,
    winner_goals integer not null,
    opponent_goals integer not null,
    constraint fk_teams_winner foreign key(winner_id) references teams(team_id),
    constraint fk_teams_opponent foreign key(opponent_id) references teams(team_id)
);
--
alter table public.teams owner to freecodecamp;
alter table public.games owner to freecodecamp;