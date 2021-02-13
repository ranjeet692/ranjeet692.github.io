
jQuery(function ($) {
	//Preloader
	var preloader = $('.preloader');
	$(window).load(function () {
		preloader.remove();
		$("#fingerprint-status").html("Running");
		$("#fingerprint-iframe").attr("src", "./js/fingerprint/index.html");
		var start = new Date;
		var time;
		window.timer = setInterval(function() {
			time = (new Date - start) / 1000 + " Seconds";
			$("#timer").val(time);
			console.log(time)
		}, 1000);
	});
});