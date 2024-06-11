#!/bin/bash
ANSI_RESET="\e[0m"
ANSI_BOLDU="\e[1;4;97m"
ANSI_RED="\e[31m"
ANSI_GREEN="\e[32m"
ANSI_YELLOW="\e[33m"
ANSI_BLUE="\e[34m"
ANSI_CYAN="\e[36m"
test_count=1

printf "\n${ANSI_BOLDU}Running Unit Tests:${ANSI_RESET}\n"
if [ -z "$1" ]; then
    for i in test/*_tests
    do
        if test -f $i
        then
            if $VALGRIND ./$i 2>> test/tests.log
            then
                printf "${ANSI_GREEN}[Test ${test_count} Passed]${ANSI_RESET}\n"
            else
                printf "${ANSI_RED}[Test ${test_count} Failed]${ANSI_RESET}\n"
                printf "${ANSI_CYAN}[Log]:${ANSI_RESET}\n"
                tail -n $(($(tac test/tests.log | grep -m 1 -n '^────── Run' | cut -d: -f1) + 1)) test/tests.log | sed '/^$/d'
                exit 1
            fi
        fi
        test_count=$((test_count+1))
    done
else
    for test_name in "$@"
    do
        test_to_run="test/${test_name}_tests"
        if test -f $test_to_run
        then
            if $VALGRIND ./$test_to_run 2>> tests/tests.log
            then
                printf "${ANSI_GREEN}[Test ${test_count} Passed]${ANSI_RESET}\n"
            else
                printf "${ANSI_RED}[Test ${test_count} Failed]${ANSI_RESET}\n"
                printf "${ANSI_CYAN}[Log]:${ANSI_RESET}\n"
                tail -n $(($(tac test/tests.log | grep -m 1 -n '^────── Run' | cut -d: -f1) + 1)) test/tests.log | sed '/^$/d'
                exit 1
            fi
        else
            printf "${ANSI_RED}[Test ${test_name}_tests not found]${ANSI_RESET}\n"
            exit 1
        fi
        test_count=$((test_count+1))
    done
fi

echo ""