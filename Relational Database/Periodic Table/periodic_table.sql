\connect periodic_table
--
alter table public.properties rename column weight to atomic_mass;
alter table public.properties rename column melting_point to melting_point_celsius;
alter table public.properties rename column boiling_point to boiling_point_celsius;
alter table public.properties alter column melting_point_celsius set not null;
alter table public.properties alter column boiling_point_celsius set not null;
alter table public.elements add constraint unique_symbol unique (symbol);
alter table public.elements add constraint unique_name unique (name);
alter table public.elements alter column symbol set not null;
alter table public.elements alter column name set not null;
alter table public.properties add constraint fk_atomic_number foreign key atomic_number references elements(atomic_number);
--
create table public.types (
    type_id serial primary key,
    type character varying(30) not null
);
--
alter table public.types owner to freecodecamp;
--
delete from public.types;
alter sequence types_type_id_seq restart with 1;
insert into public.types values (nextval('types_type_id_seq'), 'metal');
insert into public.types values (nextval('types_type_id_seq'), 'metalloid');
insert into public.types values (nextval('types_type_id_seq'), 'nonmetal');
--
alter table public.properties drop column type_id;
alter table public.properties add column type_id int;
update public.properties
set type_id = case
                 when type = 'metal' then 1
                 when type = 'metalloid' then 2
                 when type = 'nonmetal' then 3
              end;
alter table public.properties add constraint fk_type foreign key (type_id) references types(type_id);
alter table public.properties alter column type_id set not null;
--
update elements
set symbol = initcap(symbol);
--
update properties
set atomic_mass = trunc(atomic_mass,15)::double precision;
--
insert into public.elements values (9, 'f', 'fluorine');
insert into public.elements values (10, 'ne', 'neon');
insert into public.properties values (9, 'nonmetal', 18.998, -220, -188.1, 3);
insert into public.properties values (10, 'nonmetal', 20.18, -248.6, -246.1, 3);