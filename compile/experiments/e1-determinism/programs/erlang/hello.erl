-module(hello).
-export([main/0]).

main() ->
    io:format("hello ~p~n", [lists:sum([I * I || I <- lists:seq(1, 10)])]).
