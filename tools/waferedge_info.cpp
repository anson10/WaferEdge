// Prints the build and the machine: paste this next to any number you quote.
#include "waferedge/machine.hpp"

#include <cstdio>

int main() {
    std::fputs(waferedge::describe_machine().c_str(), stdout);
    return 0;
}
