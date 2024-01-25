// .ec: evo code file

// file package
mod evo:demo;

/* nihao 
woshi shuaige .
*/

// import libs
use std:{io, str:*};


// Macros with feature
#def MAX_NUM 12

// Macros with `!`
#def ADD(a, b) {
    a+b
}


// Super Class
def Animal {
    pub:
        name : str;
        age  : i32;
};

// Class Or Struct with featrue
#[extend(Animal)]
def Dog {
    pub:
        fn get_len(self: Dog) : i32 {
            self.len                    // return self.len;
        }

        fn set_name(self: Dog, name: str) : none;

    pri:
        len  : f32;
};

// append method
impl Dog {
    pub fn set_name(self: Dog, name: str) : none {
        self.name = name;
    }
};

def Vec<T> {
    data : T[];
    len  : i32;
};

// This is main function.
fn main() {
    let a = Dog {
        .name = "Alice",
        .age  = 8
    };                  // mutable Dog a
    let b := Dog {
        .name = "李华",
        .age  = 6
    };                  // const Dog b
    a.set_name("WuKong");
    
    io:fmt!("{}'s name is {}", a.name, a.get_len());

    let *pc = Str:new("hello, world!");

    scope exit {
        
    }

    
    io:fmt!("{&}", pc);
}
