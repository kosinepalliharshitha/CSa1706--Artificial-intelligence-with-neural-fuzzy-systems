animal(hen).
animal(rat).
animal(owl).
mammal(dog).
mammal(cat).
mammal(human).

is_mammal(X) :- mammal(X).
not_mammal(X) :- animal(X), \+ is_mammal(X).

check_mammal(X) :- is_mammal(X), write(X), write(' is a mammal.'), nl.
check_not_mammal(X) :- not_mammal(X), write(X), write(' is not a mammal.'), nl.
