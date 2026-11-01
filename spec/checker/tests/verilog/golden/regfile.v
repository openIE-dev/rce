// rce emit verilog (spec 10.5, spec/proposals/emit-verilog.md): the step function
// of this program as one synthesizable Verilog-2005 module. One event per clock:
// when ev_valid is high at a rising edge of clk, the step on (ev_tag, ev_payload)
// is taken; out_* hold its outcome from that edge, and the state registers change
// only when out_applied is 1. rst (synchronous, active high) loads the initial state.
// rce-hdl/1 {"emit_count_width":3,"emit_slots":6,"emit_type_width":2,"emit_types":["int","bool","Mode"],"emit_value_width":13,"error_width":1,"errors":[],"events":[{"fields":[{"name":"i","offset":0,"type":"usize","width":64},{"name":"v","offset":64,"type":"u8","width":8}],"name":"Put"},{"fields":[],"name":"Sum"},{"fields":[{"name":"d","offset":0,"type":"u16","width":16}],"name":"Div"},{"fields":[{"name":"s","offset":0,"type":"u16","width":16}],"name":"Shift"},{"fields":[{"name":"n","offset":0,"type":"u8","width":8}],"name":"Count"},{"fields":[{"name":"x","offset":0,"type":"i8","width":8}],"name":"Neg"},{"fields":[],"name":"Toggle"}],"examples":[{"events":[{"tag":0,"values":[1,7]},{"tag":0,"values":[2,9]},{"tag":1,"values":[]}],"line":2},{"events":[{"tag":0,"values":[4,1]},{"tag":1,"values":[]}],"line":3},{"events":[{"tag":0,"values":[0,12]},{"tag":1,"values":[]},{"tag":2,"values":[0]},{"tag":6,"values":[]}],"line":4},{"events":[{"tag":0,"values":[0,12]},{"tag":1,"values":[]},{"tag":2,"values":[5]},{"tag":3,"values":[3]}],"line":5},{"events":[{"tag":3,"values":[16]},{"tag":6,"values":[]}],"line":6},{"events":[{"tag":4,"values":[3]},{"tag":4,"values":[9]}],"line":7},{"events":[{"tag":5,"values":[-128]},{"tag":6,"values":[]}],"line":8},{"events":[{"tag":5,"values":[-5]},{"tag":6,"values":[]},{"tag":6,"values":[]},{"tag":6,"values":[]}],"line":9}],"format":"rce-hdl/1","kinds":{"Mode":["Idle","Busy"]},"module":"rce_File","payload_width":72,"regions":[{"fields":[{"name":"regs","offset":0,"type":{"array":"u8","len":4},"width":32},{"name":"acc","offset":32,"type":"u16","width":16},{"name":"mode","offset":48,"type":{"kind":"Mode"},"width":1}],"name":"File"}],"state_width":49,"tag_width":3,"trap_value_width":65,"trap_width":4,"traps":[{"kind":"index"},{"kind":"overflow","type":"u16"},{"kind":"text","text":"division by zero"},{"kind":"text","text":"shift amount out of range"},{"kind":"overflow","type":"u8"},{"kind":"text","text":"loop bound"},{"kind":"overflow","type":"i8"},{"kind":"text","text":"no match arm"}]}
module rce_File (
    input  wire clk,
    input  wire rst,
    input  wire ev_valid,
    input  wire [2:0] ev_tag,
    input  wire [71:0] ev_payload,
    output reg  out_valid,
    output reg  out_applied,
    output reg  [0:0] out_error,
    output reg  [3:0] out_trap,
    output reg  [64:0] out_trap_value,
    output reg  [2:0] out_emit_count,
    output reg  [11:0] out_emit_type,
    output reg  [77:0] out_emit_value,
    output wire [48:0] state
);
    // the state registers, first field at bit 0 of `state`
    reg [31:0] s0_0; // File.regs: [u8; 4]
    reg [31:0] nx0_0;
    reg [15:0] s0_1; // File.acc: u16
    reg [15:0] nx0_1;
    reg [0:0] s0_2; // File.mode: Mode
    reg [0:0] nx0_2;
    assign state = {s0_2, s0_1, s0_0};
    // the step's working registers
    reg [0:0] k; // kept: the event does not apply
    reg [0:0] kerr; // the declared error of a `fail`, plus one
    reg [3:0] ktrap; // the trap, plus one
    reg signed [64:0] kval; // the trap's value
    reg [2:0] ecnt; // values emitted
    reg [0:0] sel0; // a step of File was taken
    reg signed [8:0] l6; // regs
    reg signed [8:0] l7; // regs
    reg signed [8:0] l8; // regs
    reg signed [8:0] l9; // regs
    reg signed [16:0] l10; // acc
    reg [0:0] l11; // mode
    reg signed [64:0] l12; // i
    reg signed [8:0] l13; // v
    reg signed [64:0] l14; // held value
    reg signed [8:0] l15; // regs
    reg signed [8:0] l16; // regs
    reg signed [8:0] l17; // regs
    reg signed [8:0] l18; // regs
    reg signed [16:0] l19; // acc
    reg [0:0] l20; // mode
    reg signed [8:0] l21; // held value
    reg signed [8:0] l22; // held value
    reg signed [8:0] l23; // held value
    reg signed [8:0] l24; // held value
    reg [0:0] r25; // returned from total
    reg signed [12:0] l26; // result of total
    reg signed [8:0] l27; // xs
    reg signed [8:0] l28; // xs
    reg signed [8:0] l29; // xs
    reg signed [8:0] l30; // xs
    reg signed [12:0] l31; // s
    reg signed [8:0] l32; // held value
    reg signed [8:0] l33; // held value
    reg signed [8:0] l34; // held value
    reg signed [8:0] l35; // held value
    reg [0:0] b36; // break
    reg [0:0] c37; // continue
    reg signed [8:0] l38; // x
    reg signed [9:0] t39;
    reg signed [8:0] l40; // x
    reg signed [10:0] t41;
    reg signed [8:0] l42; // x
    reg signed [11:0] t43;
    reg signed [8:0] l44; // x
    reg signed [12:0] t45;
    reg signed [8:0] l46; // regs
    reg signed [8:0] l47; // regs
    reg signed [8:0] l48; // regs
    reg signed [8:0] l49; // regs
    reg signed [16:0] l50; // acc
    reg [0:0] l51; // mode
    reg signed [16:0] l52; // d
    reg signed [17:0] t53;
    reg signed [8:0] l54; // regs
    reg signed [8:0] l55; // regs
    reg signed [8:0] l56; // regs
    reg signed [8:0] l57; // regs
    reg signed [16:0] l58; // acc
    reg [0:0] l59; // mode
    reg signed [16:0] l60; // s
    reg signed [16:0] t61; // shift amount
    reg [15:0] t62;
    reg signed [8:0] l63; // regs
    reg signed [8:0] l64; // regs
    reg signed [8:0] l65; // regs
    reg signed [8:0] l66; // regs
    reg signed [16:0] l67; // acc
    reg [0:0] l68; // mode
    reg signed [8:0] l69; // n
    reg signed [5:0] l70; // i
    reg [0:0] b71; // break
    reg [0:0] c72; // continue
    reg signed [17:0] t73;
    reg signed [17:0] t74;
    reg signed [2:0] t75;
    reg signed [17:0] t76;
    reg signed [3:0] t77;
    reg signed [17:0] t78;
    reg signed [4:0] t79;
    reg signed [17:0] t80;
    reg signed [5:0] t81;
    reg signed [8:0] l82; // regs
    reg signed [8:0] l83; // regs
    reg signed [8:0] l84; // regs
    reg signed [8:0] l85; // regs
    reg signed [16:0] l86; // acc
    reg [0:0] l87; // mode
    reg signed [7:0] l88; // x
    reg signed [8:0] t89;
    reg signed [7:0] l90; // y
    reg signed [8:0] l91; // regs
    reg signed [8:0] l92; // regs
    reg signed [8:0] l93; // regs
    reg signed [8:0] l94; // regs
    reg signed [16:0] l95; // acc
    reg [0:0] l96; // mode
    reg [0:0] l97; // held value
    reg [0:0] r98; // returned from flip
    reg [0:0] l99; // result of flip
    reg [0:0] l100; // m
    reg [0:0] l101; // held value
    reg [0:0] m102; // a match arm was taken
    reg [1:0] et0; // the kind of emitted value 0
    reg [12:0] ev0; // emitted value 0
    reg [1:0] et1; // the kind of emitted value 1
    reg [12:0] ev1; // emitted value 1
    reg [1:0] et2; // the kind of emitted value 2
    reg [12:0] ev2; // emitted value 2
    reg [1:0] et3; // the kind of emitted value 3
    reg [12:0] ev3; // emitted value 3
    reg [1:0] et4; // the kind of emitted value 4
    reg [12:0] ev4; // emitted value 4
    reg [1:0] et5; // the kind of emitted value 5
    reg [12:0] ev5; // emitted value 5
    always @* begin
        nx0_0 = s0_0;
        nx0_1 = s0_1;
        nx0_2 = s0_2;
        k = 1'd0;
        kerr = 1'd0;
        ktrap = 4'd0;
        kval = 65'd0;
        ecnt = 3'd0;
        sel0 = 1'd0;
        l6 = 9'd0;
        l7 = 9'd0;
        l8 = 9'd0;
        l9 = 9'd0;
        l10 = 17'd0;
        l11 = 1'd0;
        l12 = 65'd0;
        l13 = 9'd0;
        l14 = 65'd0;
        l15 = 9'd0;
        l16 = 9'd0;
        l17 = 9'd0;
        l18 = 9'd0;
        l19 = 17'd0;
        l20 = 1'd0;
        l21 = 9'd0;
        l22 = 9'd0;
        l23 = 9'd0;
        l24 = 9'd0;
        r25 = 1'd0;
        l26 = 13'd0;
        l27 = 9'd0;
        l28 = 9'd0;
        l29 = 9'd0;
        l30 = 9'd0;
        l31 = 13'd0;
        l32 = 9'd0;
        l33 = 9'd0;
        l34 = 9'd0;
        l35 = 9'd0;
        b36 = 1'd0;
        c37 = 1'd0;
        l38 = 9'd0;
        t39 = 10'd0;
        l40 = 9'd0;
        t41 = 11'd0;
        l42 = 9'd0;
        t43 = 12'd0;
        l44 = 9'd0;
        t45 = 13'd0;
        l46 = 9'd0;
        l47 = 9'd0;
        l48 = 9'd0;
        l49 = 9'd0;
        l50 = 17'd0;
        l51 = 1'd0;
        l52 = 17'd0;
        t53 = 18'd0;
        l54 = 9'd0;
        l55 = 9'd0;
        l56 = 9'd0;
        l57 = 9'd0;
        l58 = 17'd0;
        l59 = 1'd0;
        l60 = 17'd0;
        t61 = 17'd0;
        t62 = 16'd0;
        l63 = 9'd0;
        l64 = 9'd0;
        l65 = 9'd0;
        l66 = 9'd0;
        l67 = 17'd0;
        l68 = 1'd0;
        l69 = 9'd0;
        l70 = 6'd0;
        b71 = 1'd0;
        c72 = 1'd0;
        t73 = 18'd0;
        t74 = 18'd0;
        t75 = 3'd0;
        t76 = 18'd0;
        t77 = 4'd0;
        t78 = 18'd0;
        t79 = 5'd0;
        t80 = 18'd0;
        t81 = 6'd0;
        l82 = 9'd0;
        l83 = 9'd0;
        l84 = 9'd0;
        l85 = 9'd0;
        l86 = 17'd0;
        l87 = 1'd0;
        l88 = 8'd0;
        t89 = 9'd0;
        l90 = 8'd0;
        l91 = 9'd0;
        l92 = 9'd0;
        l93 = 9'd0;
        l94 = 9'd0;
        l95 = 17'd0;
        l96 = 1'd0;
        l97 = 1'd0;
        r98 = 1'd0;
        l99 = 1'd0;
        l100 = 1'd0;
        l101 = 1'd0;
        m102 = 1'd0;
        et0 = 2'd0;
        ev0 = 13'd0;
        et1 = 2'd0;
        ev1 = 13'd0;
        et2 = 2'd0;
        ev2 = 13'd0;
        et3 = 2'd0;
        ev3 = 13'd0;
        et4 = 2'd0;
        ev4 = 13'd0;
        et5 = 2'd0;
        ev5 = 13'd0;
        // step File on Put(i: i, v: v) (line 39)
        if (!k && !sel0) begin
            l6 = $signed({1'b0, s0_0[7:0]});
            l7 = $signed({1'b0, s0_0[15:8]});
            l8 = $signed({1'b0, s0_0[23:16]});
            l9 = $signed({1'b0, s0_0[31:24]});
            l10 = $signed({1'b0, s0_1[15:0]});
            l11 = s0_2[0:0];
            l12 = $signed({1'b0, ev_payload[63:0]});
            l13 = $signed({1'b0, ev_payload[71:64]});
            if (((ev_tag == 3'd0))) begin
                sel0 = 1'b1;
                l14 = l12;
                if (!k && (l14 < 2'sd0 || l14 >= 4'sd4)) begin k = 1'b1; ktrap = 1; kval = l14; end
                if (l14 == 2'sd0) begin
                    l6 = l13;
                end
                if (l14 == 2'sd1) begin
                    l7 = l13;
                end
                if (l14 == 3'sd2) begin
                    l8 = l13;
                end
                if (l14 == 3'sd3) begin
                    l9 = l13;
                end
                nx0_0[7:0] = l6;
                nx0_0[15:8] = l7;
                nx0_0[23:16] = l8;
                nx0_0[31:24] = l9;
                nx0_1[15:0] = l10;
                nx0_2[0:0] = l11;
            end
        end
        // step File on Sum (line 42)
        if (!k && !sel0) begin
            l15 = $signed({1'b0, s0_0[7:0]});
            l16 = $signed({1'b0, s0_0[15:8]});
            l17 = $signed({1'b0, s0_0[23:16]});
            l18 = $signed({1'b0, s0_0[31:24]});
            l19 = $signed({1'b0, s0_1[15:0]});
            l20 = s0_2[0:0];
            if ((ev_tag == 3'd1)) begin
                sel0 = 1'b1;
                l21 = l15;
                l22 = l16;
                l23 = l17;
                l24 = l18;
                r25 = 1'b0;
                l26 = 2'sd0;
                l27 = l21;
                l28 = l22;
                l29 = l23;
                l30 = l24;
                l31 = 2'sd0;
                l32 = l27;
                l33 = l28;
                l34 = l29;
                l35 = l30;
                b36 = 1'b0;
                l38 = l32;
                c37 = 1'b0;
                if (!k && (l38 < 2'sd0 || l38 > 17'sd65535)) begin k = 1'b1; ktrap = 2; kval = l38; end
                t39 = l31 + l38;
                if (!k && (t39 < 2'sd0 || t39 > 17'sd65535)) begin k = 1'b1; ktrap = 2; kval = t39; end
                l31 = t39;
                l40 = l33;
                c37 = 1'b0;
                if (!k && (l40 < 2'sd0 || l40 > 17'sd65535)) begin k = 1'b1; ktrap = 2; kval = l40; end
                t41 = l31 + l40;
                if (!k && (t41 < 2'sd0 || t41 > 17'sd65535)) begin k = 1'b1; ktrap = 2; kval = t41; end
                l31 = t41;
                l42 = l34;
                c37 = 1'b0;
                if (!k && (l42 < 2'sd0 || l42 > 17'sd65535)) begin k = 1'b1; ktrap = 2; kval = l42; end
                t43 = l31 + l42;
                if (!k && (t43 < 2'sd0 || t43 > 17'sd65535)) begin k = 1'b1; ktrap = 2; kval = t43; end
                l31 = t43;
                l44 = l35;
                c37 = 1'b0;
                if (!k && (l44 < 2'sd0 || l44 > 17'sd65535)) begin k = 1'b1; ktrap = 2; kval = l44; end
                t45 = l31 + l44;
                if (!k && (t45 < 2'sd0 || t45 > 17'sd65535)) begin k = 1'b1; ktrap = 2; kval = t45; end
                l31 = t45;
                l26 = l31;
                r25 = 1'b1;
                l19 = l26;
                et0 = 0; ev0 = l19;
                ecnt = ecnt + 1'b1;
                et1 = 1; ev1 = (l19 > 5'sd10);
                ecnt = ecnt + 1'b1;
                nx0_0[7:0] = l15;
                nx0_0[15:8] = l16;
                nx0_0[23:16] = l17;
                nx0_0[31:24] = l18;
                nx0_1[15:0] = l19;
                nx0_2[0:0] = l20;
            end
        end
        // step File on Div(d: d) (line 47)
        if (!k && !sel0) begin
            l46 = $signed({1'b0, s0_0[7:0]});
            l47 = $signed({1'b0, s0_0[15:8]});
            l48 = $signed({1'b0, s0_0[23:16]});
            l49 = $signed({1'b0, s0_0[31:24]});
            l50 = $signed({1'b0, s0_1[15:0]});
            l51 = s0_2[0:0];
            l52 = $signed({1'b0, ev_payload[15:0]});
            if (((ev_tag == 3'd2))) begin
                sel0 = 1'b1;
                if (!k && (l52 == 2'sd0)) begin k = 1'b1; ktrap = 3; end
                if (!(l52 == 2'sd0)) t53 = l50 / l52;
                if (!k && (t53 < 2'sd0 || t53 > 17'sd65535)) begin k = 1'b1; ktrap = 2; kval = t53; end
                l50 = t53;
                nx0_0[7:0] = l46;
                nx0_0[15:8] = l47;
                nx0_0[23:16] = l48;
                nx0_0[31:24] = l49;
                nx0_1[15:0] = l50;
                nx0_2[0:0] = l51;
            end
        end
        // step File on Shift(s: s) (line 50)
        if (!k && !sel0) begin
            l54 = $signed({1'b0, s0_0[7:0]});
            l55 = $signed({1'b0, s0_0[15:8]});
            l56 = $signed({1'b0, s0_0[23:16]});
            l57 = $signed({1'b0, s0_0[31:24]});
            l58 = $signed({1'b0, s0_1[15:0]});
            l59 = s0_2[0:0];
            l60 = $signed({1'b0, ev_payload[15:0]});
            if (((ev_tag == 3'd3))) begin
                sel0 = 1'b1;
                if (!k && (l60 < 2'sd0 || l60 >= 6'sd16)) begin k = 1'b1; ktrap = 4; end
                t61 = l60;
                t62 = l58 << t61[6:0];
                l58 = $signed({1'b0, t62});
                nx0_0[7:0] = l54;
                nx0_0[15:8] = l55;
                nx0_0[23:16] = l56;
                nx0_0[31:24] = l57;
                nx0_1[15:0] = l58;
                nx0_2[0:0] = l59;
            end
        end
        // step File on Count(n: n) (line 53)
        if (!k && !sel0) begin
            l63 = $signed({1'b0, s0_0[7:0]});
            l64 = $signed({1'b0, s0_0[15:8]});
            l65 = $signed({1'b0, s0_0[23:16]});
            l66 = $signed({1'b0, s0_0[31:24]});
            l67 = $signed({1'b0, s0_1[15:0]});
            l68 = s0_2[0:0];
            l69 = $signed({1'b0, ev_payload[7:0]});
            if (((ev_tag == 3'd4))) begin
                sel0 = 1'b1;
                l70 = 2'sd0;
                b71 = 1'b0;
                if ((l70 < l69)) begin
                    c72 = 1'b0;
                    t73 = l67 + 2'sd1;
                    if (!k && (t73 < 2'sd0 || t73 > 17'sd65535)) begin k = 1'b1; ktrap = 2; kval = t73; end
                    l67 = t73;
                    l70 = 2'sd1;
                end else begin
                    b71 = 1'b1;
                end
                if (!k && !b71) begin
                    if ((l70 < l69)) begin
                        c72 = 1'b0;
                        t74 = l67 + 2'sd1;
                        if (!k && (t74 < 2'sd0 || t74 > 17'sd65535)) begin k = 1'b1; ktrap = 2; kval = t74; end
                        l67 = t74;
                        t75 = l70 + 2'sd1;
                        if (!k && (t75 < 2'sd0 || t75 > 9'sd255)) begin k = 1'b1; ktrap = 5; kval = t75; end
                        l70 = t75;
                        if ((l70 == 3'sd3)) begin
                            b71 = 1'b1;
                        end
                    end else begin
                        b71 = 1'b1;
                    end
                end
                if (!k && !b71) begin
                    if ((l70 < l69)) begin
                        c72 = 1'b0;
                        t76 = l67 + 2'sd1;
                        if (!k && (t76 < 2'sd0 || t76 > 17'sd65535)) begin k = 1'b1; ktrap = 2; kval = t76; end
                        l67 = t76;
                        t77 = l70 + 2'sd1;
                        if (!k && (t77 < 2'sd0 || t77 > 9'sd255)) begin k = 1'b1; ktrap = 5; kval = t77; end
                        l70 = t77;
                        if ((l70 == 3'sd3)) begin
                            b71 = 1'b1;
                        end
                    end else begin
                        b71 = 1'b1;
                    end
                end
                if (!k && !b71) begin
                    if ((l70 < l69)) begin
                        c72 = 1'b0;
                        t78 = l67 + 2'sd1;
                        if (!k && (t78 < 2'sd0 || t78 > 17'sd65535)) begin k = 1'b1; ktrap = 2; kval = t78; end
                        l67 = t78;
                        t79 = l70 + 2'sd1;
                        if (!k && (t79 < 2'sd0 || t79 > 9'sd255)) begin k = 1'b1; ktrap = 5; kval = t79; end
                        l70 = t79;
                        if ((l70 == 3'sd3)) begin
                            b71 = 1'b1;
                        end
                    end else begin
                        b71 = 1'b1;
                    end
                end
                if (!k && !b71) begin
                    if ((l70 < l69)) begin
                        c72 = 1'b0;
                        t80 = l67 + 2'sd1;
                        if (!k && (t80 < 2'sd0 || t80 > 17'sd65535)) begin k = 1'b1; ktrap = 2; kval = t80; end
                        l67 = t80;
                        t81 = l70 + 2'sd1;
                        if (!k && (t81 < 2'sd0 || t81 > 9'sd255)) begin k = 1'b1; ktrap = 5; kval = t81; end
                        l70 = t81;
                        if ((l70 == 3'sd3)) begin
                            b71 = 1'b1;
                        end
                    end else begin
                        b71 = 1'b1;
                    end
                end
                if (!k && !b71) begin
                    if ((l70 < l69)) begin
                        if (!k) begin k = 1'b1; ktrap = 6; end
                    end else begin
                        b71 = 1'b1;
                    end
                end
                if (ecnt == 0) begin et0 = 0; ev0 = l70; end
                if (ecnt == 1) begin et1 = 0; ev1 = l70; end
                if (ecnt == 2) begin et2 = 0; ev2 = l70; end
                ecnt = ecnt + 1'b1;
                nx0_0[7:0] = l63;
                nx0_0[15:8] = l64;
                nx0_0[23:16] = l65;
                nx0_0[31:24] = l66;
                nx0_1[15:0] = l67;
                nx0_2[0:0] = l68;
            end
        end
        // step File on Neg(x: x) (line 62)
        if (!k && !sel0) begin
            l82 = $signed({1'b0, s0_0[7:0]});
            l83 = $signed({1'b0, s0_0[15:8]});
            l84 = $signed({1'b0, s0_0[23:16]});
            l85 = $signed({1'b0, s0_0[31:24]});
            l86 = $signed({1'b0, s0_1[15:0]});
            l87 = s0_2[0:0];
            l88 = $signed(ev_payload[7:0]);
            if (((ev_tag == 3'd5))) begin
                sel0 = 1'b1;
                t89 = -l88;
                if (!k && (t89 < (-9'sd128) || t89 > 8'sd127)) begin k = 1'b1; ktrap = 7; kval = t89; end
                l90 = t89;
                if (ecnt == 0) begin et0 = 0; ev0 = l90; end
                if (ecnt == 1) begin et1 = 0; ev1 = l90; end
                if (ecnt == 2) begin et2 = 0; ev2 = l90; end
                if (ecnt == 3) begin et3 = 0; ev3 = l90; end
                ecnt = ecnt + 1'b1;
                if (ecnt == 1) begin et1 = 2; ev1 = l87; end
                if (ecnt == 2) begin et2 = 2; ev2 = l87; end
                if (ecnt == 3) begin et3 = 2; ev3 = l87; end
                if (ecnt == 4) begin et4 = 2; ev4 = l87; end
                ecnt = ecnt + 1'b1;
                nx0_0[7:0] = l82;
                nx0_0[15:8] = l83;
                nx0_0[23:16] = l84;
                nx0_0[31:24] = l85;
                nx0_1[15:0] = l86;
                nx0_2[0:0] = l87;
            end
        end
        // step File on Toggle (line 67)
        if (!k && !sel0) begin
            l91 = $signed({1'b0, s0_0[7:0]});
            l92 = $signed({1'b0, s0_0[15:8]});
            l93 = $signed({1'b0, s0_0[23:16]});
            l94 = $signed({1'b0, s0_0[31:24]});
            l95 = $signed({1'b0, s0_1[15:0]});
            l96 = s0_2[0:0];
            if ((ev_tag == 3'd6)) begin
                sel0 = 1'b1;
                l97 = l96;
                r98 = 1'b0;
                l99 = 1'd0;
                l100 = l97;
                l101 = l100;
                m102 = 1'b0;
                if (!k && !m102) begin
                    if ((l101 == 1'd0)) begin
                        m102 = 1'b1;
                        l99 = 1'd1;
                        r98 = 1'b1;
                    end
                end
                if (!k && !m102) begin
                    if ((l101 == 1'd1)) begin
                        m102 = 1'b1;
                        l99 = 1'd0;
                        r98 = 1'b1;
                    end
                end
                if (!m102) begin
                    if (!k) begin k = 1'b1; ktrap = 8; end
                end
                l96 = l99;
                if (ecnt == 0) begin et0 = 2; ev0 = l96; end
                if (ecnt == 1) begin et1 = 2; ev1 = l96; end
                if (ecnt == 2) begin et2 = 2; ev2 = l96; end
                if (ecnt == 3) begin et3 = 2; ev3 = l96; end
                if (ecnt == 4) begin et4 = 2; ev4 = l96; end
                if (ecnt == 5) begin et5 = 2; ev5 = l96; end
                ecnt = ecnt + 1'b1;
                nx0_0[7:0] = l91;
                nx0_0[15:8] = l92;
                nx0_0[23:16] = l93;
                nx0_0[31:24] = l94;
                nx0_1[15:0] = l95;
                nx0_2[0:0] = l96;
            end
        end
    end
    always @(posedge clk) begin
        if (rst) begin
            s0_0 <= 32'h00000000;
            s0_1 <= 16'h0000;
            s0_2 <= 1'h0;
            out_valid <= 1'd0;
            out_applied <= 1'd0;
            out_error <= 1'd0;
            out_trap <= 4'd0;
            out_trap_value <= 65'd0;
            out_emit_count <= 3'd0;
            out_emit_type <= 12'd0;
            out_emit_value <= 78'd0;
        end else begin
            out_valid <= ev_valid;
            if (ev_valid) begin
                out_applied <= !k;
                out_error <= kerr;
                out_trap <= ktrap;
                out_trap_value <= kval;
                out_emit_count <= k ? 3'd0 : ecnt;
                out_emit_type <= {et5, et4, et3, et2, et1, et0};
                out_emit_value <= {ev5, ev4, ev3, ev2, ev1, ev0};
                if (!k) begin
                    s0_0 <= nx0_0;
                    s0_1 <= nx0_1;
                    s0_2 <= nx0_2;
                end
            end
        end
    end
endmodule
