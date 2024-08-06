#!/usr/local/bin/lua

require("luaevo")

local arr = {
    {1, 3, 4},
    {2, 5, 8},
    {3, 6, 9}
}
local ts = evo.Tensor(arr)
ts:dump2(1)