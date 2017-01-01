// Basic Example of Unconditional Handwriting Generation.
var sketch = function( p ) { 
  "use strict";

  // variables we need for this demo
  var dx, dy; // offsets of the pen strokes, in pixels
  var pen, prev_pen; // keep track of whether pen is touching paper
  var x, y; // absolute coordinates on the screen of where the pen is
  var temperature = 0.25; // controls the amount of uncertainty of the model
  var rnn_state; // store the hidden states of rnn's neurons
  var pdf; // store all the parameters of a mixture-density distribution

  var has_started = false; // set to true after user starts writing.
  var epsilon = 2.0; // to ignore data from user's pen staying in one spot.

  var screen_width, screen_height; // stores the browser's dimensions

  var predict_len = 10; // predict the next N steps constantly

  var restart = function() {
    // reinitialize variables before calling p5.js setup.

    // make sure we enforce some minimum size of our demo
    screen_width = Math.max(window.innerWidth, 480);
    screen_height = Math.max(window.innerHeight, 320)/1.0;

    x = 50; // start drawing 50 pixels from the left side of the canvas
    y = screen_height/2; // start drawing from the middle of the canvas

    has_started = false;

    // initialize the scale factor for the model. Bigger -> large outputs
    Model.set_scale_factor(10.0);

    // initialize pen's states to zero.
    [dx, dy, prev_pen] = Model.zero_input(); // the pen's states

    // initialize the rnn's initial states to zero
    rnn_state = Model.random_state();

  };

  p.setup = function() {
    restart(); // initialize variables for this demo
    p.createCanvas(screen_width, screen_height);
    p.frameRate(60);
    p.background(255, 255, 255, 255);
    p.fill(255, 255, 255, 255);
  };

  var predict_path = function() {
    var temp_state = Model.copy_state(rnn_state); // create a copy
    var temp_dx, temp_dy, temp_pen, temp_prev_pen=prev_pen;
    var temp_x = x;
    var temp_y = y;
    var c = p.color(255, 165, 0, 48);

    for (var i=0; i<predict_len; i++) {

      // get the parameters of the probability distribution (pdf) from hidden state
      pdf = Model.get_pdf(temp_state);
      // sample the next pen's states from our probability distribution
      [temp_dx, temp_dy, temp_pen] = Model.sample(pdf, temperature);

      // only draw on the paper if the pen is touching the paper
      if (temp_prev_pen == 0) {
        p.stroke(c); // orangy colour
        p.strokeWeight(0.5); // nice thick line
        // draw line connecting prev point to current point.
        p.line(temp_x, temp_y, temp_x+temp_dx, temp_y+temp_dy);
      }

      // update the absolute coordinates from the offsets
      temp_x += temp_dx;
      temp_y += temp_dy;

      // update the previous pen's state to the current one we just sampled
      temp_prev_pen = temp_pen;

      // update state
      temp_state = Model.update([temp_dx, temp_dy, temp_prev_pen], temp_state);

    }
  };

  p.draw = function() {

    // record pen drawing from user:
    if (p.mouseIsPressed) { // pen is touching the paper
      if (has_started == false) { // first time anything is written
        has_started = true;
        x = p.mouseX;
        y = p.mouseY;
      }
      var dx0 = p.mouseX-x; // candidate for dx
      var dy0 = p.mouseY-y; // candidate for dy
      if (dx0*dx0+dy0*dy0 > epsilon*epsilon) { // only if pen is not in same area
        dx = dx0;
        dy = dy0;
        pen = 0;

        if (prev_pen == 0) {
          p.stroke(255,165,0);
          p.strokeWeight(1.0); // nice thick line
          p.line(x, y, x+dx, y+dy); // draw line connecting prev point to current point.
        }

        // update the absolute coordinates from the offsets
        x += dx;
        y += dy;

        // using the previous pen states, and hidden state, get next hidden state 
        rnn_state = Model.update([dx, dy, prev_pen], rnn_state);
      }
    } else { // pen is above the paper
      pen = 1;
      if (pen !== prev_pen) {
        // using the previous pen states, and hidden state, get next hidden state 
        rnn_state = Model.update([dx, dy, prev_pen], rnn_state);
      }
    }

    if (has_started) {
      // predict a sample for the next possible path at every frame update.
      predict_path(rnn_state);
    }

    // update the previous pen's state to the current one we just sampled
    prev_pen = pen;

    /*
    if (p.frameCount % 8 == 0) {
      p.background(255, 255, 255, 16); // fade out a bit.
      p.fill(255, 255, 255, 32);
    }
    */

  };

};
var custom_p5 = new p5(sketch, 'sketch');
