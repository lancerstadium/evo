#define EVO_LUABIND
#include "../cpp/Evo.hpp"
// #include <lua.hpp>
// #include "LuaBridge/LuaBridge.h"

using namespace luabridge;

extern "C" int luaopen_luaevo(lua_State* L) {

    Evo::internal_mdl = Evo::Model::from(internal_model_init());
    getGlobalNamespace(L)
        .beginNamespace("evo")
        .beginClass<Evo::Tensor>("Tensor")
        .addConstructor<void(*)()>()
        .addConstructor<void(*)(LuaRef)>()
        .addFunction("dump", static_cast<void (Evo::Tensor::*)()>(&Evo::Tensor::dump))
        .addFunction("dump2", static_cast<void (Evo::Tensor::*)(int)>(&Evo::Tensor::dump))
        .addFunction("__tostring", &Evo::Tensor::toString)
        .addFunction("__add", &Evo::Tensor::operator+)
        .addFunction("__mul", &Evo::Tensor::operator*)
        .endClass()
        .endNamespace();

    
    return 0;
}
