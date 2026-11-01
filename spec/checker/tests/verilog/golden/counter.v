// rce emit verilog (spec 10.5, spec/proposals/emit-verilog.md): the step function
// of this program as one synthesizable Verilog-2005 module. One event per clock:
// when ev_valid is high at a rising edge of clk, the step on (ev_tag, ev_payload)
// is taken; out_* hold its outcome from that edge, and the state registers change
// only when out_applied is 1. rst (synchronous, active high) loads the initial state.
// rce-hdl/1 {"emit_count_width":1,"emit_slots":0,"emit_type_width":1,"emit_types":[],"emit_value_width":1,"error_width":1,"errors":["Underflow"],"events":[{"fields":[],"name":"Inc"},{"fields":[],"name":"Dec"}],"examples":[{"events":[{"tag":0,"values":[]},{"tag":0,"values":[]},{"tag":1,"values":[]}],"line":2},{"events":[{"tag":1,"values":[]}],"line":3},{"events":[{"tag":1,"values":[]},{"tag":0,"values":[]}],"line":4}],"format":"rce-hdl/1","kinds":{},"module":"rce_Counter","payload_width":1,"regions":[{"fields":[{"name":"n","offset":0,"type":"u64","width":64}],"name":"Counter"}],"state_width":64,"tag_width":1,"trap_value_width":66,"trap_width":1,"traps":[{"kind":"overflow","type":"u64"}]}
module rce_Counter (
    input  wire clk,
    input  wire rst,
    input  wire ev_valid,
    input  wire [0:0] ev_tag,
    input  wire [0:0] ev_payload,
    output reg  out_valid,
    output reg  out_applied,
    output reg  [0:0] out_error,
    output reg  [0:0] out_trap,
    output reg  [65:0] out_trap_value,
    output reg  [0:0] out_emit_count,
    output reg  [0:0] out_emit_type,
    output reg  [0:0] out_emit_value,
    output wire [63:0] state
);
    // the state registers, first field at bit 0 of `state`
    reg [63:0] s0_0; // Counter.n: u64
    reg [63:0] nx0_0;
    assign state = {s0_0};
    // the step's working registers
    reg [0:0] k; // kept: the event does not apply
    reg [0:0] kerr; // the declared error of a `fail`, plus one
    reg [0:0] ktrap; // the trap, plus one
    reg signed [65:0] kval; // the trap's value
    reg [0:0] ecnt; // values emitted
    reg [0:0] sel0; // a step of Counter was taken
    reg signed [64:0] l6; // n
    reg signed [65:0] t7;
    reg signed [64:0] l8; // n
    reg signed [65:0] t9;
    always @* begin
        nx0_0 = s0_0;
        k = 1'd0;
        kerr = 1'd0;
        ktrap = 1'd0;
        kval = 66'd0;
        ecnt = 1'd0;
        sel0 = 1'd0;
        l6 = 65'd0;
        t7 = 66'd0;
        l8 = 65'd0;
        t9 = 66'd0;
        // step Counter on Inc (line 14)
        if (!k && !sel0) begin
            l6 = $signed({1'b0, s0_0[63:0]});
            if ((ev_tag == 1'd0)) begin
                sel0 = 1'b1;
                t7 = l6 + 2'sd1;
                if (!k && (t7 < 2'sd0 || t7 > 65'sd18446744073709551615)) begin k = 1'b1; ktrap = 1; kval = t7; end
                l6 = t7;
                nx0_0[63:0] = l6;
            end
        end
        // step Counter on Dec (line 15)
        if (!k && !sel0) begin
            l8 = $signed({1'b0, s0_0[63:0]});
            if ((ev_tag == 1'd1)) begin
                sel0 = 1'b1;
                t9 = l8 - 2'sd1;
                if (!k && (t9 < 2'sd0 || t9 > 65'sd18446744073709551615)) begin k = 1'b1; kerr = 1; end
                l8 = t9;
                nx0_0[63:0] = l8;
            end
        end
    end
    always @(posedge clk) begin
        if (rst) begin
            s0_0 <= 64'h0000000000000000;
            out_valid <= 1'd0;
            out_applied <= 1'd0;
            out_error <= 1'd0;
            out_trap <= 1'd0;
            out_trap_value <= 66'd0;
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
                end
            end
        end
    end
endmodule
