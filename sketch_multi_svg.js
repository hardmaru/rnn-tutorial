// variables we need for this demo
var dx, dy; // offsets of the pen strokes, in pixels
var pen, prev_pen; // keep track of whether pen is touching paper
var x, y; // absolute coordinates on the screen of where the pen is
var temperature = 0.50; // controls the amount of uncertainty of the model
var rnn_state; // store the hidden states of rnn's neurons
var pdf; // store all the parameters of a mixture-density distribution

var has_started = false; // set to true after user starts writing.
var epsilon = 4.0; // to ignore data from user's pen staying in one spot.

var screen_width, screen_height; // stores the browser's dimensions

var predict_len = 5; // predict the next N steps constantly

// create a copy for predictor
var temp_state;
var temp_dx, temp_dy, temp_pen, temp_prev_pen;
var temp_x = x;
var temp_y = y;
var c_predict;

var restart = function() {
  // reinitialize variables before calling p5.js setu

  // make sure we enforce some minimum size of our demo
  screen_width = Math.max(window.innerWidth, 480);
  screen_height = Math.max(window.innerHeight, 320)/1.0;

  c_predict = color(241, 148, 138, 64); // set color of prediction.

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

var copy_rnn_state = function() {
  temp_state = Model.copy_state(rnn_state);
  temp_prev_pen = prev_pen;
  temp_x = x;
  temp_y = y;
};

var update_rnn_state = function() {
  rnn_state = Model.update([dx, dy, prev_pen], rnn_state);
  copy_rnn_state();
};

setup = function() {
  restart(); // initialize variables for this demo
  createCanvas(screen_width, screen_height, SVG);
  frameRate(60);
  background(255, 255, 255, 255);
  fill(255, 255, 255, 255);
};

var predict_path = function() {

  for (var i=0; i<predict_len; i++) {

    // get the parameters of the probability distribution (pdf) from hidden state
    pdf = Model.get_pdf(temp_state);
    // sample the next pen's states from our probability distribution
    [temp_dx, temp_dy, temp_pen] = Model.sample(pdf, temperature);

    // only draw on the paper if the pen is touching the paper
    if (temp_prev_pen == 0) {
      stroke(c_predict); // orangy colour
      strokeWeight(1.0); // nice thick line
      // draw line connecting prev point to current point.
      line(temp_x, temp_y, temp_x+temp_dx, temp_y+temp_dy);
    }

    // update the absolute coordinates from the offsets
    temp_x += temp_dx;
    temp_y += temp_dy;

    // update the previous pen's state to the current one we just sampled
    temp_prev_pen = temp_pen;

    // update state
    temp_state = Model.update([temp_dx, temp_dy, temp_prev_pen], temp_state);

    // reset when drawing off right side of screen
    if (temp_x > screen_width) {
      copy_rnn_state();
    }

  }
}

draw = function() {

  // record pen drawing from user:
  if (mouseIsPressed) { // pen is touching the paper
    if (has_started == false) { // first time anything is written
      has_started = true;
      x = mouseX;
      y = mouseY;
      copy_rnn_state();
    }
    if (mouseX > screen_width-50) {
      // save svg file.
      noLoop();
      save();
    }
    var dx0 = mouseX-x; // candidate for dx
    var dy0 = mouseY-y; // candidate for dy
    if (dx0*dx0+dy0*dy0 > epsilon*epsilon) { // only if pen is not in same area
      dx = dx0;
      dy = dy0;
      pen = 0;

      if (prev_pen == 0) {
        stroke(241, 148, 138);
        strokeWeight(1.5); // nice thick line
        line(x, y, x+dx, y+dy); // draw line connecting prev point to current point.
      }

      // update the absolute coordinates from the offsets
      x += dx;
      y += dy;

      // using the previous pen states, and hidden state, get next hidden state 
      update_rnn_state();
    }
  } else { // pen is above the paper
    pen = 1;
    if (pen !== prev_pen) {
      // using the previous pen states, and hidden state, get next hidden state 
      update_rnn_state();
    }
  }

  if (has_started) {
    // predict a sample for the next possible path at every frame update.
    predict_path(rnn_state);
  }

  // update the previous pen's state to the current one we just sampled
  prev_pen = pen;

  /*
  if (frameCount % 8 == 0) {
    background(255, 255, 255, 16); // fade out a bit.
    fill(255, 255, 255, 32);
  }
  */

};