#!/bin/bash
assert() {
    expected="$1"
    input="$2"

    ./evoc "$input" > tmp.s
    cc tmp.s -o tmp
    ./tmp
    actual="$?"

    if [ "$actual" = "$expected" ]; then
        echo "$input => $actual"
    else
        echo "$input => $expected expected, but got $actual"
        exit 1
    fi
}

assert 0 '0;'
assert 42 '42;'
assert 21 '5+20-4;'
assert 41 '12+34-5;'
assert 48 '5 + 21+ 23 -1;'

assert 47 '5+6 * 7;'
assert 15 '5*(9-6);'
assert 4 '(3+5)/2;'
assert 4 '-(3+5)/2 +8;'
assert 43 '+(5+47)-+(1*9);'
# assert -8 '-(3+5)+3*+5-3*+5;'  # -8 excepted but got 248
assert 10 '- -10;'
assert 10 '- - +10;'

assert 0 '0==1;'
assert 1 '42==42;'
assert 1 '0!=1;'
assert 0 '42!=42;'

assert 1 '0<1;'
assert 0 '1<1;'
assert 0 '2<1;'
assert 1 '0<=1;'
assert 1 '1<=1;'
assert 0 '2<=1;'

assert 1 '1>0;'
assert 0 '1>1;'
assert 0 '1>2;'
assert 1 '1>=0;'
assert 1 '1>=1;'
assert 0 '1>=2;'

# assert 3 '1; 2; 3;'

# assert 3 'a=3; a;'
# assert 8 'a=3; z=5; a+z;'
# assert 6 'a=b=3; a+b;'
# assert 3 'foo=3; foo;'
# assert 8 'foo123=3; bar=5; foo123+bar;'

assert 0 '{ return 0; }'
assert 42 '{ return 42; }'
assert 21 '{ return 5+20-4; }'
assert 41 '{ return  12 + 34 - 5 ; }'
assert 47 '{ return 5+6*7; }'
assert 15 '{ return 5*(9-6); }'
assert 4 '{ return (3+5)/2; }'
assert 10 '{ return -10+20; }'
assert 10 '{ return - -10; }'
assert 10 '{ return - - +10; }'

assert 0 '{ return 0==1; }'
assert 1 '{ return 42==42; }'
assert 1 '{ return 0!=1; }'
assert 0 '{ return 42!=42; }'

assert 1 '{ return 0<1; }'
assert 0 '{ return 1<1; }'
assert 0 '{ return 2<1; }'
assert 1 '{ return 0<=1; }'
assert 1 '{ return 1<=1; }'
assert 0 '{ return 2<=1; }'

assert 1 '{ return 1>0; }'
assert 0 '{ return 1>1; }'
assert 0 '{ return 1>2; }'
assert 1 '{ return 1>=0; }'
assert 1 '{ return 1>=1; }'
assert 0 '{ return 1>=2; }'

# assert 3 '{ a=3; return a; }'
# assert 8 '{ a=3; z=5; return a+z; }'

# assert 3 '{ a=3; return a; }'
# assert 8 '{ a=3; z=5; return a+z; }'
# assert 6 '{ a=b=3; return a+b; }'
# assert 3 '{ foo=3; return foo; }'
# assert 8 '{ foo123=3; bar=5; return foo123+bar; }'

# assert 1 '{ return 1; 2; 3; }'
# assert 2 '{ 1; return 2; 3; }'
# assert 3 '{ 1; 2; return 3; }'

# assert 3 '{ {1; {2;} return 3;} }'

echo "OK"