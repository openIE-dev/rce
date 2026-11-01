// rce_tb.v: the test bench of `rce emit verilog` (spec 10.5,
// spec/proposals/emit-verilog.md EV-6), written in Verilog-2005 with the
// file tasks of IEEE 1364-2005 section 17.2. bench.py, next to this file,
// gives the widths of the module under test as macros and the event
// stream as the plusarg stream=PATH.
//
// The stream (rce-hdl-stream/1) is a sequence of records. A record that
// starts with the byte 0 resets the module: the bench prints `R STATE`
// with the state registers after the reset. A record that starts with the
// byte 1 is one event: TB bytes of the tag and PB bytes of the payload,
// least significant byte first. The bench drives the event bus for one
// clock and prints `S APPLIED ERROR TRAP TRAPVALUE EMITCOUNT EMITTYPES
// EMITVALUES STATE`, each in hexadecimal, from the module's outputs after
// that clock edge. The bench computes nothing: bench.py turns the dumps
// into the trace of each example.
`timescale 1ns/1ps
module rce_tb;
    reg clk;
    reg rst;
    reg ev_valid;
    reg [`TW-1:0] ev_tag;
    reg [`PW-1:0] ev_payload;
    reg [8*`TB-1:0] tbuf;
    reg [8*`PB-1:0] pbuf;
    wire out_valid;
    wire out_applied;
    wire [`ERW-1:0] out_error;
    wire [`TRW-1:0] out_trap;
    wire [`TVW-1:0] out_trap_value;
    wire [`NCW-1:0] out_emit_count;
    wire [`ETW-1:0] out_emit_type;
    wire [`EVW-1:0] out_emit_value;
    wire [`SW-1:0] state;
    `RCE_TOP dut (
        .clk(clk),
        .rst(rst),
        .ev_valid(ev_valid),
        .ev_tag(ev_tag),
        .ev_payload(ev_payload),
        .out_valid(out_valid),
        .out_applied(out_applied),
        .out_error(out_error),
        .out_trap(out_trap),
        .out_trap_value(out_trap_value),
        .out_emit_count(out_emit_count),
        .out_emit_type(out_emit_type),
        .out_emit_value(out_emit_value),
        .state(state)
    );
    reg [8*4096-1:0] path;
    integer fd;
    integer c;
    integer i;
    initial clk = 1'b0;
    always #5 clk = ~clk;
    initial begin
        rst = 1'b1;
        ev_valid = 1'b0;
        ev_tag = 0;
        ev_payload = 0;
        if (!$value$plusargs("stream=%s", path)) begin
            $display("E no stream: give +stream=PATH");
            $finish;
        end
        fd = $fopen(path, "rb");
        if (fd == 0) begin
            $display("E the stream does not open");
            $finish;
        end
        c = $fgetc(fd);
        while (c != -1) begin
            @(negedge clk);
            if (c == 0) begin
                rst = 1'b1;
                ev_valid = 1'b0;
                @(posedge clk);
                #1;
                rst = 1'b0;
                $display("R %h", state);
            end else begin
                tbuf = 0;
                pbuf = 0;
                for (i = 0; i < `TB; i = i + 1) begin
                    c = $fgetc(fd);
                    tbuf[8*i +: 8] = c[7:0];
                end
                for (i = 0; i < `PB; i = i + 1) begin
                    c = $fgetc(fd);
                    pbuf[8*i +: 8] = c[7:0];
                end
                ev_tag = tbuf[`TW-1:0];
                ev_payload = pbuf[`PW-1:0];
                ev_valid = 1'b1;
                @(posedge clk);
                #1;
                ev_valid = 1'b0;
                $display("S %h %h %h %h %h %h %h %h", out_applied, out_error, out_trap, out_trap_value,
                         out_emit_count, out_emit_type, out_emit_value, state);
            end
            c = $fgetc(fd);
        end
        $fclose(fd);
        $finish;
    end
endmodule
