% Персонажи
character(v).
character(johnny_silverhand).
character(judy_alvarez).
character(panam_palmer).
character(jackie_welles).
character(rogue_amendiares).
character(evelyn_parker).
character(takemura).
character(alt_cunningham).
character(meredith_stout).
character(dexter_deshawn).
character(misty).
character(wakako_okada).
character(kerry_eurodyne).
character(viktor_vector).
character(lizzy_wizzy).
character(mr_hands).
character(ozob).
character(t_bug).
character(smasher).

% Локации
location(night_city).
location(watson).
location(westbrook).
location(pacifica).
location(santo_domingo).
location(heywood).
location(city_center).
location(badlands).
location(afterlife).

% Оружие
weapon(katana).
weapon(smart_pistol).
weapon(power_rifle).
weapon(tech_sniper).
weapon(shotgun).
weapon(monowire).
weapon(pistol).

% Корпорации
corporation(arasaka).
corporation(militech).
corporation(kang_tao).
corporation(biotechnica).
corporation(trauma_team).
corporation(netwatch).

% Киберимпланты
cyberware(mantis_blades).
cyberware(gorilla_arms).
cyberware(sandevistan).
cyberware(kerensikov).
cyberware(berserk).
cyberware(neural_link).
cyberware(cyberarm).

% Фракции
faction(aldecaldos).
faction(valentinos).
faction(moxes).
faction(tyger_claws).
faction(animals).
faction(netrunners).

% --------------------------------------------------------

% Персонаж находится в локации
character_in_location(v, night_city).
character_in_location(johnny_silverhand, night_city).
character_in_location(judy_alvarez, watson).
character_in_location(panam_palmer, badlands).
character_in_location(kerry_eurodyne, westbrook).
character_in_location(rogue_amendiares, afterlife).
character_in_location(takemura, night_city).
character_in_location(wakako_okada, westbrook).
character_in_location(viktor_vector, watson).
character_in_location(meredith_stout, night_city).

% Персонаж обладает оружием
has_weapon(v, katana).
has_weapon(v, smart_pistol).
has_weapon(johnny_silverhand, power_rifle).
has_weapon(panam_palmer, tech_sniper).
has_weapon(jackie_welles, shotgun).
has_weapon(rogue_amendiares, pistol).
has_weapon(dexter_deshawn, pistol).
has_weapon(ozob, monowire).

% Персонаж имеет киберимплант
cyberware_installed(v, mantis_blades).
cyberware_installed(v, sandevistan).
cyberware_installed(judy_alvarez, neural_link).
cyberware_installed(panam_palmer, kerensikov).
cyberware_installed(johnny_silverhand, cyberarm).

% Персонаж работает на корпорацию
works_for(takemura, arasaka).
works_for(meredith_stout, militech).
works_for(smasher, arasaka).
works_for(alt_cunningham, netwatch).

% Персонаж является членом фракции
member_of(panam_palmer, aldecaldos).
member_of(jackie_welles, valentinos).
member_of(judy_alvarez, moxes).
member_of(ozob, animals).
member_of(t_bug, netrunners).

% Персонаж является другом другого персонажа
friend_of(v, judy_alvarez).
friend_of(v, panam_palmer).
friend_of(v, kerry_eurodyne).
friend_of(v, jackie_welles).
friend_of(v, viktor_vector).

% Персонаж является врагом другого персонажа
enemy_of(v, arasaka).
enemy_of(v, smasher).
enemy_of(johnny_silverhand, arasaka).
enemy_of(rogue_amendiares, arasaka).

% --------------------------------------------------------

% Правила

% Персонаж имеет киберимпланты
has_cyberware(Character) :-
    cyberware_installed(Character, _).

% Персонаж является врагом корпорации Арасака
is_enemy_of_arasaka(Character) :-
    enemy_of(Character, arasaka).

% Персонаж является другом V
is_friend_of_v(Character) :-
    friend_of(v, Character).

% Персонаж работает на корпорацию
is_corporate_agent(Character) :-
    works_for(Character, Corporation),
    corporation(Corporation).

% Персонаж находится в Найт-Сити
is_in_night_city(Character) :-
    character_in_location(Character, night_city).

% Персонажи имеют связь (дружба или вражда)
has_relationship(Character1, Character2) :-
    friend_of(Character1, Character2);
    friend_of(Character2, Character1);
    enemy_of(Character1, Character2);
    enemy_of(Character2, Character1).

% Персонаж обладает мощным киберимплантом
has_powerful_cyberware(Character) :-
    cyberware_installed(Character, sandevistan);
    cyberware_installed(Character, mantis_blades).

% Персонаж вооружен опасным оружием
is_heavily_armed(Character) :-
    has_weapon(Character, Weapon),
    weapon(Weapon),
    (Weapon = tech_sniper; Weapon = monowire; Weapon = shotgun).

% Персонаж является членом банды
is_gang_member(Character) :-
    member_of(Character, Faction),
    faction(Faction).
