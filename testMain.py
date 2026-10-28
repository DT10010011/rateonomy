import ROIGUI
import ROIRating


def main():
	roi_app = ROIGUI.ROIApp()
	roi_app.mainloop()
	ROIRating.printpayback(roi_app)


if __name__ == "__main__":
	main()
