
mod hello (
    submod1 = "path/to/mod1"
    submod2 = "path/to/mod2"
)

use (
    texc = "github.com/lancerstadium/texc"
    core = "../core/core.h"
    io   = std.io.*
)

#def MC dog
#udf MC

type MyInt : int

enum AnimalType {
    CAT
    DOG
    BIRD
}

struct Animal {
    *Obj                        // 匿名结构体，表示继承Obj类型的属性和方法
    type       : AnimalType
    MC##_name  : str
    MC##_no    : i32

    fn getNo() : i32 {
        self.dog_no;
    }
    fn getName(num : i32) : str {}
}

// fn Animal.getName(T)(num : i32) : str {
//     self.dog_name;
// }

fn main(argc : int, argv : str) : i32 {
    d1 := Animal (              // var d1 = Animal
        dog_name = "John"
        dog_no   = 12
    )

    arr1 := []i32 (1, 2, 3, 4)              // 数组类型：栈分配，边界检查
    slice1 : &[]i32 = &arr1[1:3]            // 数组引用
    vector2 := ()any (4, "xiexie", 8)       // 容器类型：堆分配，边界检查

    d2 ~= Animal( "Alice", 13 ) // const d2 = Animal
    d1.getName(wuhu)

    ~Animal(d1)
}