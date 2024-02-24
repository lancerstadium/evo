
mod hello (
    submod1 = "path/to/mod1"
    submod2 = "path/to/mod2"
)

use (
    texc = "github.com/lancerstadium/texc"
    core = "../core/core.h"
    io   = std.io.*
)

#def MC Dog
#udf MC

type MyInt : i32

enum AnimalType {
    CAT
    DOG
    BIRD
}

struct Animal {
    Obj                        // 匿名结构体，表示继承Obj类型的属性和方法
    Type       : AnimalType
    MC##_name  : str
    MC##_no    : i32

    fn getNo() : i32 {
        self.dog_no;
    }
    fn getName(num : i32) : str 
}

// fn Animal.getName(T)(num : i32) : str {
//     self.dog_name;
// }

fn main(argc : i32, argv : str) : i32 {
    d1 := Animal (              
        Dog_name = "John"
        Dog_no   = 12
    )

    arr1 := []i32 (1, 2, 3, 4)              // 数组类型：栈分配，边界检查
    vector2 := vec[]any (4, "xiexie", 8)    // 容器类型：堆分配，边界检查
    

    d2 ~= Animal( "Alice", 13 ) // const d2 = Animal
    d1.getName(wuhu)

    ~Animal(d1)
}