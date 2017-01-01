// handwriting mdn-lstm model ported to JS

if (typeof module != "undefined") {
}

var Model = {};

(function(global) {
  "use strict";

  // init (import weights from weights.js)

  var string_to_uint8array = function(b64encoded) {
    var u8 = new Uint8Array(atob(b64encoded).split("").map(function(c) {
        return c.charCodeAt(0); }));
    return u8;
  }

  var uintarray_to_string = function(u8) {
    var b64encoded = btoa(String.fromCharCode.apply(null, u8));
    return b64encoded;
  };

  function encode_array(raw_array) {
    var i;
    var N = raw_array.length;
    var x = [];
    for (i=0;i<N;i++) {
      x.push(raw_array[i]+127);
    }
    var u8 = new Uint8Array(x);
    return uintarray_to_string(u8);
  }

  function decode_string(b64encoded) {
    var i;
    var raw_array = string_to_uint8array(b64encoded);
    var N = raw_array.length;
    var x = [];
    for (i=0;i<N;i++) {
      x.push(raw_array[i]-127);
    }
    return x;
  }

  function encode_2d_array(raw_2d_array) {
    var i;
    var N = raw_2d_array.length;
    var x = [];
    var temp;
    for (i=0;i<N;i++) {
      temp = encode_array(raw_2d_array[i]);
      x.push(temp);
    }
    return x;
  }

  function decode_2d_string(raw_2d_array) {
    var i;
    var N = raw_2d_array.length;
    var x = [];
    var temp;
    for (i=0;i<N;i++) {
      temp = decode_string(raw_2d_array[i]);
      x.push(temp);
    }
    return x;
  }

  var bdata = JSON.parse(lstm_data);

  var raw_output_w = decode_2d_string(bdata[0]);
  var raw_output_b = decode_string(bdata[1]);
  var raw_LSTM_Wxh = decode_2d_string(bdata[2]);
  var raw_LSTM_Whh = decode_2d_string(bdata[3]);
  var raw_LSTM_bias = decode_string(bdata[4]);

  var scale_output_w = 1.5;
  var scale_output_b = 2.5;
  var scale_LSTM_Wxh = 2.0;
  var scale_LSTM_Whh = 0.8;
  var scale_LSTM_bias = 1.5;

  var output_w = nj.array(raw_output_w);
  var output_b = nj.array(raw_output_b);
  var LSTM_Wxh = nj.array(raw_LSTM_Wxh);
  var LSTM_Whh = nj.array(raw_LSTM_Whh);
  var LSTM_bias = nj.array(raw_LSTM_bias);

  output_w = output_w.divide(127);
  output_w = output_w.multiply(scale_output_w);
  output_b = output_b.divide(127);
  output_b = output_b.multiply(scale_output_b);
  LSTM_Wxh = LSTM_Wxh.divide(127);
  LSTM_Wxh = LSTM_Wxh.multiply(scale_LSTM_Wxh);
  LSTM_Whh = LSTM_Whh.divide(127);
  LSTM_Whh = LSTM_Whh.multiply(scale_LSTM_Whh);
  LSTM_bias = LSTM_bias.divide(127);
  LSTM_bias = LSTM_bias.multiply(scale_LSTM_bias);

  // settings
  var num_units=500;
  var N_mixture=20;
  var input_size=3;
  var W_full=nj.concatenate([LSTM_Wxh.T, LSTM_Whh.T]).T; // training size
  var bias=LSTM_bias;
  var forget_bias=1.0;
  var h_w = output_w;
  var h_b = output_b;

  var scale_factor = 6.0;

  var zero_state = function() {
    return [nj.zeros(num_units), nj.zeros(num_units)];
  };

  var random_state = function() {
    var std_ = 0.50;
    var h = nj.zeros(num_units);
    var c = nj.zeros(num_units);
    var i = 0;
    for(i=0;i<num_units;i++) {
      h.set(i, randn(0, std_));
      c.set(i, randn(0, std_));
    }
    return [h, c];
  };

  var copy_state = function(state) {
    var h = state[0].clone();
    var c = state[1].clone();
    return [h, c];
  };

  var zero_input = function() {
    return [0, 0, 0];
  };

  var random_input = function() {
    var std_ = 0.1;
    var x = nj.zeros(input_size);
    var pen_s = 0;
    if (randf(0, 1) > 0.9) pen_s = 1;
    x.set(0, randn(0, std_));
    x.set(1, randn(0, std_));
    x.set(2, pen_s);
    return x;
  };

  var update = function(x_, s) {
    var x = nj.zeros(input_size);
    x.set(0, x_[0]/scale_factor);
    x.set(1, x_[1]/scale_factor);
    x.set(2, x_[2]);
    var h = s[0];
    var c = s[1];
    var concat = nj.concatenate([x, h]);
    var hidden = nj.dot(concat, W_full);
    hidden = nj.add(hidden, bias);

    var i=nj.sigmoid(hidden.slice([0*num_units, 1*num_units]));
    var g=nj.tanh(hidden.slice([1*num_units, 2*num_units]));
    var f=nj.sigmoid(nj.add(hidden.slice([2*num_units, 3*num_units]), forget_bias));
    var o=nj.sigmoid(hidden.slice([3*num_units, 4*num_units]));

    var new_c = nj.add(nj.multiply(c, f), nj.multiply(g, i));
    var new_h = nj.multiply(nj.tanh(new_c), o);

    return [new_h, new_c];
  }

  var get_pdf = function(s) {
    var h = s[0];
    var NOUT = N_mixture;
    var z=nj.add(nj.dot(h, h_w), h_b);
    var z_eos = nj.sigmoid(z.slice([0, 1]));
    var z_pi = z.slice([1+NOUT*0, 1+NOUT*1]);
    var z_mu1 = z.slice([1+NOUT*1, 1+NOUT*2]);
    var z_mu2 = z.slice([1+NOUT*2, 1+NOUT*3]);
    var z_sigma1 = nj.exp(z.slice([1+NOUT*3, 1+NOUT*4]));
    var z_sigma2 = nj.exp(z.slice([1+NOUT*4, 1+NOUT*5]));
    var z_corr = nj.tanh(z.slice([1+NOUT*5, 1+NOUT*6]));
    z_pi = nj.subtract(z_pi, z_pi.max());
    z_pi = nj.softmax(z_pi);

    return [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_eos];
  };

  var sample_pi_idx = function(z_pi) {
    var x = randf(0, 1);
    var N = N_mixture;
    var accumulate = 0;
    var i = 0;
    for (i=0;i<N;i++) {
      accumulate += z_pi.get(i);
      if (accumulate >= x) {
        return i;
      }
    }
    console.log('error sampling pi index');
    return -1;
  };

  var sample_eos = function(z_eos) {
    // eos = 1 if random.random() < o_eos[0][0] else 0
    var eos = 0;
    if (randf(0, 1) < z_eos.get(0)) {
      eos = 1;
    }
    return eos;
  }

  /*
  def adjust_temp(pi_pdf, temp):
    pi_pdf = np.log(pi_pdf) / temp
    pi_pdf -= pi_pdf.max()
    pi_pdf = np.exp(pi_pdf)
    pi_pdf /= pi_pdf.sum()
    return pi_pdf
  */

  var adjust_temp = function(z_old, temp) {
    var z = nj.array(z_old);
    var i;
    var x;
    //console.log("before="+z_old.get(0));
    for (i=z.shape[0]-1;i>=0;i--) {
      x = z.get(i);
      x = Math.log(x) / temp;
      z.set(i, x);
    }
    x = z.max();
    z = nj.subtract(z, x);
    z = nj.exp(z);
    x = z.sum();
    z = nj.divide(z, x);
    //console.log("after="+z.get(0));
    return z;
  };

  var sample = function(z, temperature) {
    // z is [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_eos]
    // returns [x, y, eos]
    var temp=0.65;
    if (typeof(temperature) === "number") {
      temp = temperature;
    }
    var z_0 = adjust_temp(z[0], temp);
    var z_6 = nj.array(z[6]);
    //var z6 = Math.exp(Math.log(z_6.get(0))/temp);
    //z_6.set(0, z6);
    var idx = sample_pi_idx(z_0);
    var mu1 = z[1].get(idx);
    var mu2 = z[2].get(idx);
    var sigma1 = z[3].get(idx)*temp;
    var sigma2 = z[4].get(idx)*temp;
    var corr = z[5].get(idx);
    var eos = sample_eos(z_6);
    var delta = birandn(mu1, mu2, sigma1, sigma2, corr);
    return [delta[0]*scale_factor, delta[1]*scale_factor, eos];
  }

  // Random numbers util (from https://github.com/karpathy/recurrentjs)
  var return_v = false;
  var v_val = 0.0;
  var gaussRandom = function() {
    if(return_v) { 
      return_v = false;
      return v_val; 
    }
    var u = 2*Math.random()-1;
    var v = 2*Math.random()-1;
    var r = u*u + v*v;
    if(r == 0 || r > 1) return gaussRandom();
    var c = Math.sqrt(-2*Math.log(r)/r);
    v_val = v*c; // cache this
    return_v = true;
    return u*c;
  }
  var randf = function(a, b) { return Math.random()*(b-a)+a; };
  var randi = function(a, b) { return Math.floor(Math.random()*(b-a)+a); };
  var randn = function(mu, std){ return mu+gaussRandom()*std; };
  // from http://www.math.grin.edu/~mooret/courses/math336/bivariate-normal.html
  var birandn = function(mu1, mu2, std1, std2, rho) {
    var z1 = randn(0, 1);
    var z2 = randn(0, 1);
    var x = Math.sqrt(1-rho*rho)*std1*z1 + rho*std1*z2 + mu1;
    var y = std2*z2 + mu2;
    return [x, y];
  };

  var set_scale_factor = function(scale) {
    scale_factor = scale;
  };

  global.zero_state = zero_state;
  global.zero_input = zero_input;
  global.random_state = random_state;
  global.copy_state = copy_state;
  global.random_input = random_input;
  global.update = update;
  global.get_pdf = get_pdf;
  global.randf = randf;
  global.randi = randi;
  global.randn = randn;
  global.birandn = birandn;
  global.sample = sample;
  global.set_scale_factor = set_scale_factor;

})(Model);
(function(lib) {
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    //window.jsfeat = lib; // in ordinary browser attach library to window
  } else {
    module.exports = lib; // in nodejs
  }
})(Model);
