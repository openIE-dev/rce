// rce emit verilog (spec 10.5, spec/proposals/emit-verilog.md): the step function
// of this program as one synthesizable Verilog-2005 module. One event per clock:
// when ev_valid is high at a rising edge of clk, the step on (ev_tag, ev_payload)
// is taken; out_* hold its outcome from that edge, and the state registers change
// only when out_applied is 1. rst (synchronous, active high) loads the initial state.
// rce-hdl/1 {"emit_count_width":1,"emit_slots":0,"emit_type_width":1,"emit_types":[],"emit_value_width":1,"error_width":1,"errors":["Quit"],"events":[{"fields":[{"name":"code","offset":0,"type":"u8","width":8},{"name":"down","offset":8,"type":"bool","width":1}],"name":"Key"}],"examples":[{"events":[{"tag":0,"values":[65,true]}],"line":2},{"events":[{"tag":0,"values":[27,true]}],"line":3}],"format":"rce-hdl/1","kinds":{},"module":"rce_Pad","payload_width":9,"regions":[{"fields":[{"name":"presses","offset":0,"type":"u8","width":8}],"name":"Pad"}],"state_width":8,"tag_width":1,"trap_value_width":10,"trap_width":1,"traps":[{"kind":"overflow","type":"u8"}]}
module rce_Pad (
    input  wire clk,
    input  wire rst,
    input  wire ev_valid,
    input  wire [0:0] ev_tag,
    input  wire [8:0] ev_payload,
    output reg  out_valid,
    output reg  out_applied,
    output reg  [0:0] out_error,
    output reg  [0:0] out_trap,
    output reg  [9:0] out_trap_value,
    output reg  [0:0] out_emit_count,
    output reg  [0:0] out_emit_type,
    output reg  [0:0] out_emit_value,
    output wire [7:0] state
);
    // the state registers, first field at bit 0 of `state`
    reg [7:0] s0_0; // Pad.presses: u8
    reg [7:0] nx0_0;
    assign state = {s0_0};
    // the step's working registers
    reg [0:0] k; // kept: the event does not apply
    reg [0:0] kerr; // the declared error of a `fail`, plus one
    reg [0:0] ktrap; // the trap, plus one
    reg signed [9:0] kval; // the trap's value
    reg [0:0] ecnt; // values emitted
    reg [0:0] sel0; // a step of Pad was taken
    reg signed [8:0] l6; // presses
    reg signed [8:0] l7; // presses
    reg signed [9:0] t8;
    reg signed [8:0] l9; // presses
    always @* begin
        nx0_0 = s0_0;
        k = 1'd0;
        kerr = 1'd0;
        ktrap = 1'd0;
        kval = 10'd0;
        ecnt = 1'd0;
        sel0 = 1'd0;
        l6 = 9'd0;
        l7 = 9'd0;
        t8 = 10'd0;
        l9 = 9'd0;
        // step Pad on Key(code: 27, down: true) (line 20)
        if (!k && !sel0) begin
            l6 = $signed({1'b0, s0_0[7:0]});
            if (((ev_tag == 1'd0) && ($signed({1'b0, ev_payload[7:0]}) == 6'sd27) && (ev_payload[8] == 1'b1))) begin
                sel0 = 1'b1;
                if (!k) begin k = 1'b1; kerr = 1; end
                nx0_0[7:0] = l6;
            end
        end
        // step Pad on Key(code: _, down: true) (line 23)
        if (!k && !sel0) begin
            l7 = $signed({1'b0, s0_0[7:0]});
            if (((ev_tag == 1'd0) && (ev_payload[8] == 1'b1))) begin
                sel0 = 1'b1;
                if ((l7 < 9'sd255)) begin
                    t8 = l7 + 2'sd1;
                    if (!k && (t8 < 2'sd0 || t8 > 9'sd255)) begin k = 1'b1; ktrap = 1; kval = t8; end
                    l7 = t8;
                end
                nx0_0[7:0] = l7;
            end
        end
        // step Pad on Key(code: _, down: false) (line 27)
        if (!k && !sel0) begin
            l9 = $signed({1'b0, s0_0[7:0]});
            if (((ev_tag == 1'd0) && (ev_payload[8] == 1'b0))) begin
                sel0 = 1'b1;
                nx0_0[7:0] = l9;
            end
        end
    end
    always @(posedge clk) begin
        if (rst) begin
            s0_0 <= 8'h00;
            out_valid <= 1'd0;
            out_applied <= 1'd0;
            out_error <= 1'd0;
            out_trap <= 1'd0;
            out_trap_value <= 10'd0;
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
