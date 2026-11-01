// rce emit verilog (spec 10.5, spec/proposals/emit-verilog.md): the step function
// of this program as one synthesizable Verilog-2005 module. One event per clock:
// when ev_valid is high at a rising edge of clk, the step on (ev_tag, ev_payload)
// is taken; out_* hold its outcome from that edge, and the state registers change
// only when out_applied is 1. rst (synchronous, active high) loads the initial state.
// rce-hdl/1 {"emit_count_width":1,"emit_slots":0,"emit_type_width":1,"emit_types":[],"emit_value_width":1,"error_width":1,"errors":["Empty"],"events":[{"fields":[{"name":"k","offset":0,"type":"u8","width":8}],"name":"Add"},{"fields":[{"name":"k","offset":0,"type":"u8","width":8}],"name":"Take"},{"fields":[{"name":"f","offset":0,"type":"u16","width":16}],"name":"Scale"}],"examples":[{"events":[{"tag":0,"values":[200]},{"tag":0,"values":[55]}],"line":2},{"events":[{"tag":0,"values":[200]},{"tag":0,"values":[56]},{"tag":0,"values":[1]}],"line":3},{"events":[{"tag":1,"values":[1]}],"line":4},{"events":[{"tag":0,"values":[3]},{"tag":2,"values":[100]},{"tag":2,"values":[2]}],"line":5}],"format":"rce-hdl/1","kinds":{},"module":"rce_Acc","payload_width":16,"regions":[{"fields":[{"name":"total","offset":0,"type":"u16","width":16}],"name":"Acc"}],"state_width":16,"tag_width":2,"trap_value_width":34,"trap_width":1,"traps":[{"kind":"overflow","type":"u16"}]}
module rce_Acc (
    input  wire clk,
    input  wire rst,
    input  wire ev_valid,
    input  wire [1:0] ev_tag,
    input  wire [15:0] ev_payload,
    output reg  out_valid,
    output reg  out_applied,
    output reg  [0:0] out_error,
    output reg  [0:0] out_trap,
    output reg  [33:0] out_trap_value,
    output reg  [0:0] out_emit_count,
    output reg  [0:0] out_emit_type,
    output reg  [0:0] out_emit_value,
    output wire [15:0] state
);
    // the state registers, first field at bit 0 of `state`
    reg [15:0] s0_0; // Acc.total: u16
    reg [15:0] nx0_0;
    assign state = {s0_0};
    // the step's working registers
    reg [0:0] k; // kept: the event does not apply
    reg [0:0] kerr; // the declared error of a `fail`, plus one
    reg [0:0] ktrap; // the trap, plus one
    reg signed [33:0] kval; // the trap's value
    reg [0:0] ecnt; // values emitted
    reg [0:0] sel0; // a step of Acc was taken
    reg signed [16:0] l6; // total
    reg signed [8:0] l7; // k
    reg signed [17:0] t8;
    reg signed [17:0] t9;
    reg signed [16:0] l10; // total
    reg signed [8:0] l11; // k
    reg signed [17:0] t12;
    reg signed [16:0] l13; // total
    reg signed [16:0] l14; // f
    reg signed [33:0] t15;
    always @* begin
        nx0_0 = s0_0;
        k = 1'd0;
        kerr = 1'd0;
        ktrap = 1'd0;
        kval = 34'd0;
        ecnt = 1'd0;
        sel0 = 1'd0;
        l6 = 17'd0;
        l7 = 9'd0;
        t8 = 18'd0;
        t9 = 18'd0;
        l10 = 17'd0;
        l11 = 9'd0;
        t12 = 18'd0;
        l13 = 17'd0;
        l14 = 17'd0;
        t15 = 34'd0;
        // step Acc on Add(k: k) (line 16)
        if (!k && !sel0) begin
            l6 = $signed({1'b0, s0_0[15:0]});
            l7 = $signed({1'b0, ev_payload[7:0]});
            if (((ev_tag == 2'd0))) begin
                sel0 = 1'b1;
                if (!k && (l7 < 2'sd0 || l7 > 17'sd65535)) begin k = 1'b1; ktrap = 1; kval = l7; end
                t8 = l6 + l7;
                if (!k && (t8 < 2'sd0 || t8 > 17'sd65535)) begin k = 1'b1; ktrap = 1; kval = t8; end
                l6 = t8;
                if ((l6 > 9'sd255)) begin
                    t9 = l6 + 17'sd65535;
                    if (!k && (t9 < 2'sd0 || t9 > 17'sd65535)) begin k = 1'b1; ktrap = 1; kval = t9; end
                    l6 = t9;
                end
                nx0_0[15:0] = l6;
            end
        end
        // step Acc on Take(k: k) (line 21)
        if (!k && !sel0) begin
            l10 = $signed({1'b0, s0_0[15:0]});
            l11 = $signed({1'b0, ev_payload[7:0]});
            if (((ev_tag == 2'd1))) begin
                sel0 = 1'b1;
                if (!k && (l11 < 2'sd0 || l11 > 17'sd65535)) begin k = 1'b1; ktrap = 1; kval = l11; end
                t12 = l10 - l11;
                if (!k && (t12 < 2'sd0 || t12 > 17'sd65535)) begin k = 1'b1; kerr = 1; end
                l10 = t12;
                nx0_0[15:0] = l10;
            end
        end
        // step Acc on Scale(f: f) (line 23)
        if (!k && !sel0) begin
            l13 = $signed({1'b0, s0_0[15:0]});
            l14 = $signed({1'b0, ev_payload[15:0]});
            if (((ev_tag == 2'd2))) begin
                sel0 = 1'b1;
                t15 = l13 * l14;
                if (!k && (t15 < 2'sd0 || t15 > 17'sd65535)) begin k = 1'b1; ktrap = 1; kval = t15; end
                l13 = t15;
                nx0_0[15:0] = l13;
            end
        end
    end
    always @(posedge clk) begin
        if (rst) begin
            s0_0 <= 16'h0000;
            out_valid <= 1'd0;
            out_applied <= 1'd0;
            out_error <= 1'd0;
            out_trap <= 1'd0;
            out_trap_value <= 34'd0;
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
