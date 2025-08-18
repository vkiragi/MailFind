import SwiftUI

struct SearchView: View {
    @State private var query = ""
    @FocusState private var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            TextField("Search emails…", text: $query)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 14))
                .focused($focused)
                .onAppear { focused = true }

            Divider().padding(.vertical, 4)

            // Placeholder results list for now
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Results will appear here.")
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(12)
        .frame(width: 400, height: 480)
    }
}
