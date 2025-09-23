#!/usr/local/bin/lua

require("luaevo")

local arr1 = {
    {1, 3, 4},
    {2, 5, 8},
}
local arr2 = {
    {2, 5, 8},
    {3, 6, 9}
}
local ts1 = evo.Tensor(arr1)
local ts2 = evo.Tensor(arr2)
print(ts1)
print(ts2)
local ts3 = ts1 + ts2
print(ts3)
local ts4 = ts1 * ts2
print(ts4)