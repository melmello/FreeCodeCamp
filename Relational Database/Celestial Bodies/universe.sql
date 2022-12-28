drop database universe;
create database universe;
--
alter database universe owner to freecodecamp;
--
\connect universe
--
create table public.galaxy (
    galaxy_id serial primary key,
    name character varying(30) not null,
    earth_distance integer,
    diameter integer,
    mass numeric(5,1),
    category text,
    is_new boolean,
    constraint galaxy_name_unique unique (name)
);
create table public.star (
    star_id serial primary key,
    galaxy_id integer not null,
    name character varying(30) not null,
    earth_distance integer,
    diameter integer,
    mass numeric(5,1),
    category text,
    is_new boolean,
    constraint fk_galaxy foreign key(galaxy_id) references galaxy(galaxy_id),
    constraint star_name_unique unique (name)
);
create table public.planet (
    planet_id serial primary key,
    star_id integer not null,
    name character varying(30) not null,
    earth_distance integer,
    diameter integer,
    mass numeric(5,1),
    category text,
    is_new boolean,
    constraint fk_star foreign key(star_id) references star(star_id),
    constraint planet_name_unique unique (name)
);
create table public.moon (
    moon_id serial primary key,
    planet_id integer not null,
    name character varying(30) not null,
    earth_distance integer,
    diameter integer,
    mass numeric(5,1),
    category text,
    is_new boolean,
    constraint fk_planet foreign key(planet_id) references planet(planet_id),
    constraint moon_name_unique unique (name)
);
create table public.satellite (
    satellite_id serial primary key,
    planet_id integer not null,
    name character varying(30) not null,
    earth_distance integer,
    diameter integer,
    mass numeric(5,1),
    category text,
    is_new boolean,
    constraint fk_planet foreign key(planet_id) references planet(planet_id),
    constraint satellite_name_unique unique (name)
);
--
alter table public.galaxy owner to freecodecamp;
alter table public.star owner to freecodecamp;
alter table public.planet owner to freecodecamp;
alter table public.moon owner to freecodecamp;
alter table public.satellite owner to freecodecamp;
--
insert into public.galaxy values (nextval('galaxy_galaxy_id_seq'), 'galaxy1', null, null, null, null, null);
insert into public.galaxy values (nextval('galaxy_galaxy_id_seq'), 'galaxy2', null, null, null, null, null);
insert into public.galaxy values (nextval('galaxy_galaxy_id_seq'), 'galaxy3', null, null, null, null, null);
insert into public.galaxy values (nextval('galaxy_galaxy_id_seq'), 'galaxy4', null, null, null, null, null);
insert into public.galaxy values (nextval('galaxy_galaxy_id_seq'), 'galaxy5', null, null, null, null, null);
insert into public.galaxy values (nextval('galaxy_galaxy_id_seq'), 'galaxy6', null, null, null, null, null);
--
insert into public.star values (nextval('star_star_id_seq'), 1, 'star1', null, null, null, null, null);
insert into public.star values (nextval('star_star_id_seq'), 2, 'star2', null, null, null, null, null);
insert into public.star values (nextval('star_star_id_seq'), 3, 'star3', null, null, null, null, null);
insert into public.star values (nextval('star_star_id_seq'), 4, 'star4', null, null, null, null, null);
insert into public.star values (nextval('star_star_id_seq'), 5, 'star5', null, null, null, null, null);
insert into public.star values (nextval('star_star_id_seq'), 6, 'star6', null, null, null, null, null);
--
insert into public.planet values (nextval('planet_planet_id_seq'), 1, 'planet1', null, null, null, null, null);
insert into public.planet values (nextval('planet_planet_id_seq'), 2, 'planet2', null, null, null, null, null);
insert into public.planet values (nextval('planet_planet_id_seq'), 3, 'planet3', null, null, null, null, null);
insert into public.planet values (nextval('planet_planet_id_seq'), 4, 'planet4', null, null, null, null, null);
insert into public.planet values (nextval('planet_planet_id_seq'), 5, 'planet5', null, null, null, null, null);
insert into public.planet values (nextval('planet_planet_id_seq'), 6, 'planet6', null, null, null, null, null);
insert into public.planet values (nextval('planet_planet_id_seq'), 1, 'planet7', null, null, null, null, null);
insert into public.planet values (nextval('planet_planet_id_seq'), 2, 'planet8', null, null, null, null, null);
insert into public.planet values (nextval('planet_planet_id_seq'), 3, 'planet9', null, null, null, null, null);
insert into public.planet values (nextval('planet_planet_id_seq'), 4, 'planet10', null, null, null, null, null);
insert into public.planet values (nextval('planet_planet_id_seq'), 5, 'planet11', null, null, null, null, null);
insert into public.planet values (nextval('planet_planet_id_seq'), 6, 'planet12', null, null, null, null, null);
--
insert into public.moon values (nextval('moon_moon_id_seq'), 1, 'moon1', null, null, null, null, null);
insert into public.moon values (nextval('moon_moon_id_seq'), 2, 'moon2', null, null, null, null, null);
insert into public.moon values (nextval('moon_moon_id_seq'), 3, 'moon3', null, null, null, null, null);
insert into public.moon values (nextval('moon_moon_id_seq'), 4, 'moon4', null, null, null, null, null);
insert into public.moon values (nextval('moon_moon_id_seq'), 5, 'moon5', null, null, null, null, null);
insert into public.moon values (nextval('moon_moon_id_seq'), 6, 'moon6', null, null, null, null, null);
insert into public.moon values (nextval('moon_moon_id_seq'), 7, 'moon7', null, null, null, null, null);
insert into public.moon values (nextval('moon_moon_id_seq'), 8, 'moon8', null, null, null, null, null);
insert into public.moon values (nextval('moon_moon_id_seq'), 9, 'moon9', null, null, null, null, null);
insert into public.moon values (nextval('moon_moon_id_seq'), 10, 'moon10', null, null, null, null, null);
insert into public.moon values (nextval('moon_moon_id_seq'), 11, 'moon11', null, null, null, null, null);
insert into public.moon values (nextval('moon_moon_id_seq'), 12, 'moon12', null, null, null, null, null);
insert into public.moon values (nextval('moon_moon_id_seq'), 1, 'moon13', null, null, null, null, null);
insert into public.moon values (nextval('moon_moon_id_seq'), 2, 'moon14', null, null, null, null, null);
insert into public.moon values (nextval('moon_moon_id_seq'), 3, 'moon15', null, null, null, null, null);
insert into public.moon values (nextval('moon_moon_id_seq'), 4, 'moon16', null, null, null, null, null);
insert into public.moon values (nextval('moon_moon_id_seq'), 5, 'moon17', null, null, null, null, null);
insert into public.moon values (nextval('moon_moon_id_seq'), 6, 'moon18', null, null, null, null, null);
insert into public.moon values (nextval('moon_moon_id_seq'), 7, 'moon19', null, null, null, null, null);
insert into public.moon values (nextval('moon_moon_id_seq'), 8, 'moon20', null, null, null, null, null);
--
insert into public.satellite values (nextval('satellite_satellite_id_seq'), 1, 'satellite18', null, null, null, null, null);
insert into public.satellite values (nextval('satellite_satellite_id_seq'), 2, 'satellite19', null, null, null, null, null);
insert into public.satellite values (nextval('satellite_satellite_id_seq'), 3, 'satellite20', null, null, null, null, null);