mod hello (
    submod1 : "path/to/mod1"
    submod2 : "path/to/mod2"
)

use (
    texc : "github.com/lancerstadium/texc"
    core : "../core/core.h"
    io   : std.io.*
)

#def MC dog
#udf MC

type MyInt int

enum AnimalType {
    CAT
    DOG
    BIRD
}

type Animal {
    *Obj                        // 匿名结构体，表示继承Obj类型的属性和方法
    type       : AnimalType
    MC##_name  : str
    MC##_no    : i32

    fn getNo() : i32 {
        self.dog_no;
    }
    fn getName(T)(T num) : str
}

fn (Animal)getName(T)(num : T) : str {
    self.dog_name;
}

fn main(argc : int, argv : str) {
    d1 := Animal (              // var d1 = Animal
        dog_name : "John"
        dog_no   : 12
    )
    d2 ~= Animal( "Alice", 13 ) // const d2 = Animal
    d1.getName(wuhu)
}