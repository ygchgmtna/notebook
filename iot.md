intel(x86)
1. esp寄存器是栈顶指针，每次push和pop操作会改变。
2. ebp是栈基地址指针，用于在函数执行过程中定位参数和局部变量，每次函数调用，会变化，在变化前会保持主调函数的ebp，在函数执行完返回前，恢复主调函数。
3. call指令会入栈它的下一条指令地址，ret指令从栈顶取出，并跳转执行。

如果局部变量中有字符串、数组等数据，然后使用了字符串拷贝、内存拷贝等操作。如果越界，会形成一个严重的后果，就是call指令入栈数据可能被覆盖，函数ret指令执行时，就不会到预想的call指令的下一条地址。
