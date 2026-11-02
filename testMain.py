import ROIGUI
import Rating


def main():
	roi_app = ROIGUI.ROIApp()
	roi_app.mainloop()
	Rating.printpayback(roi_app)


if __name__ == "__main__":
	main()
