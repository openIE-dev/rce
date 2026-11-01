// rce emit verilog (spec 10.5, spec/proposals/emit-verilog.md): the step function
// of this program as one synthesizable Verilog-2005 module. One event per clock:
// when ev_valid is high at a rising edge of clk, the step on (ev_tag, ev_payload)
// is taken; out_* hold its outcome from that edge, and the state registers change
// only when out_applied is 1. rst (synchronous, active high) loads the initial state.
// rce-hdl/1 {"emit_count_width":1,"emit_slots":0,"emit_type_width":1,"emit_types":[],"emit_value_width":1,"error_width":2,"errors":["DoorOpen","Locked"],"events":[{"fields":[],"name":"Open"},{"fields":[],"name":"Close"},{"fields":[],"name":"Lock"},{"fields":[],"name":"Unlock"}],"examples":[{"events":[{"tag":0,"values":[]},{"tag":1,"values":[]},{"tag":2,"values":[]}],"line":3},{"events":[{"tag":0,"values":[]},{"tag":2,"values":[]}],"line":4}],"format":"rce-hdl/1","kinds":{},"module":"rce_Door","payload_width":1,"regions":[{"fields":[{"name":"open","offset":0,"type":"bool","width":1},{"name":"locked","offset":1,"type":"bool","width":1}],"name":"Door"}],"state_width":2,"tag_width":2,"trap_value_width":2,"trap_width":1,"traps":[]}
module rce_Door (
    input  wire clk,
    input  wire rst,
    input  wire ev_valid,
    input  wire [1:0] ev_tag,
    input  wire [0:0] ev_payload,
    output reg  out_valid,
    output reg  out_applied,
    output reg  [1:0] out_error,
    output reg  [0:0] out_trap,
    output reg  [1:0] out_trap_value,
    output reg  [0:0] out_emit_count,
    output reg  [0:0] out_emit_type,
    output reg  [0:0] out_emit_value,
    output wire [1:0] state
);
    // the state registers, first field at bit 0 of `state`
    reg [0:0] s0_0; // Door.open: bool
    reg [0:0] nx0_0;
    reg [0:0] s0_1; // Door.locked: bool
    reg [0:0] nx0_1;
    assign state = {s0_1, s0_0};
    // the step's working registers
    reg [0:0] k; // kept: the event does not apply
    reg [1:0] kerr; // the declared error of a `fail`, plus one
    reg [0:0] ktrap; // the trap, plus one
    reg signed [1:0] kval; // the trap's value
    reg [0:0] ecnt; // values emitted
    reg [0:0] sel0; // a step of Door was taken
    reg [0:0] l6; // open
    reg [0:0] l7; // locked
    reg [0:0] l8; // open
    reg [0:0] l9; // locked
    reg [0:0] l10; // open
    reg [0:0] l11; // locked
    reg [0:0] l12; // open
    reg [0:0] l13; // locked
    reg [0:0] l14; // open
    reg [0:0] l15; // locked
    reg [0:0] l16; // open
    reg [0:0] l17; // locked
    always @* begin
        nx0_0 = s0_0;
        nx0_1 = s0_1;
        k = 1'd0;
        kerr = 2'd0;
        ktrap = 1'd0;
        kval = 2'd0;
        ecnt = 1'd0;
        sel0 = 1'd0;
        l6 = 1'd0;
        l7 = 1'd0;
        l8 = 1'd0;
        l9 = 1'd0;
        l10 = 1'd0;
        l11 = 1'd0;
        l12 = 1'd0;
        l13 = 1'd0;
        l14 = 1'd0;
        l15 = 1'd0;
        l16 = 1'd0;
        l17 = 1'd0;
        // step Door on Open (line 18)
        if (!k && !sel0) begin
            l6 = s0_0[0];
            l7 = s0_1[0];
            if ((ev_tag == 2'd0)) begin
                if (!k && l7) begin
                    sel0 = 1'b1;
                    if (!k) begin k = 1'b1; kerr = 2; end
                    nx0_0[0] = l6;
                    nx0_1[0] = l7;
                end
            end
        end
        // step Door on Open (line 20)
        if (!k && !sel0) begin
            l8 = s0_0[0];
            l9 = s0_1[0];
            if ((ev_tag == 2'd0)) begin
                sel0 = 1'b1;
                l8 = 1'b1;
                nx0_0[0] = l8;
                nx0_1[0] = l9;
            end
        end
        // step Door on Close (line 22)
        if (!k && !sel0) begin
            l10 = s0_0[0];
            l11 = s0_1[0];
            if ((ev_tag == 2'd1)) begin
                sel0 = 1'b1;
                l10 = 1'b0;
                nx0_0[0] = l10;
                nx0_1[0] = l11;
            end
        end
        // step Door on Lock (line 24)
        if (!k && !sel0) begin
            l12 = s0_0[0];
            l13 = s0_1[0];
            if ((ev_tag == 2'd2)) begin
                if (!k && l12) begin
                    sel0 = 1'b1;
                    if (!k) begin k = 1'b1; kerr = 1; end
                    nx0_0[0] = l12;
                    nx0_1[0] = l13;
                end
            end
        end
        // step Door on Lock (line 26)
        if (!k && !sel0) begin
            l14 = s0_0[0];
            l15 = s0_1[0];
            if ((ev_tag == 2'd2)) begin
                sel0 = 1'b1;
                l15 = 1'b1;
                nx0_0[0] = l14;
                nx0_1[0] = l15;
            end
        end
        // step Door on Unlock (line 28)
        if (!k && !sel0) begin
            l16 = s0_0[0];
            l17 = s0_1[0];
            if ((ev_tag == 2'd3)) begin
                sel0 = 1'b1;
                l17 = 1'b0;
                nx0_0[0] = l16;
                nx0_1[0] = l17;
            end
        end
    end
    always @(posedge clk) begin
        if (rst) begin
            s0_0 <= 1'h0;
            s0_1 <= 1'h0;
            out_valid <= 1'd0;
            out_applied <= 1'd0;
            out_error <= 2'd0;
            out_trap <= 1'd0;
            out_trap_value <= 2'd0;
            out_emit_count <= 1'd0;
            out_emit_type <= 1'd0;
            out_emit_value <= 1'd0;
        end else begin
            out_valid <= ev_valid;
            if (ev_valid) begin
                out_applied <= !k;
                out_error <= kerr;
                out_trap <= ktrap;
                out_trap_value <= kval;
                out_emit_count <= k ? 1'd0 : ecnt;
                out_emit_type <= 1'd0;
                out_emit_value <= 1'd0;
                if (!k) begin
                    s0_0 <= nx0_0;
                    s0_1 <= nx0_1;
                end
            end
        end
    end
endmodule
